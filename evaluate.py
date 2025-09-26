from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import average_precision_score
import torch.functional as F

'''
Based on code from https://github.com/usuyama/ePillID-benchmark
'''

def create_same_side_pairs(labels, is_front, is_ref=None):
    '''
    If is_ref is not None, will ensure that refs are paired withe refs and cons are paired with cons
    will return pairs such that the first ind corresponds to front side and the second ind corresponds to back side
    '''
    labels = labels if type(labels) == np.ndarray else labels.cpu().data.numpy()
    is_front = is_front if type(is_front) == np.ndarray else is_front.cpu().data.numpy()
    valid_pairs = (labels[is_front].reshape(-1,1) == labels[~is_front])
    if is_ref is not None:
        is_ref = is_ref if type(is_ref) == np.ndarray else is_ref.cpu().data.numpy()
        valid_pairs &= (is_ref[is_front].reshape(-1,1) == is_ref[~is_front])
    f_i, b_i = np.where(valid_pairs)
    return np.c_[np.argwhere(is_front)[f_i], np.argwhere(~is_front)[b_i]]

def side_aggregation(front, back, aggr_mode="max"):
    stacked_sides = torch.stack((front, back), dim=2)
    if aggr_mode == "max":
        return stacked_sides.max(dim=2)[0]
    elif aggr_mode == "min":
        return stacked_sides.min(dim=2)[0]
    elif aggr_mode == "mean":
        return stacked_sides.max(dim=2)
    else:
        raise f"Unknown aggr_mode: {aggr_mode}"
    
def aggr_ref_axis(scores, ref_labels, aggr_mode="max", sorted=False, return_ref_labels=True):
    if not sorted:
        sorted_ref_inds = ref_labels.argsort()
        scores = scores[:, sorted_ref_inds]
        ref_labels = ref_labels[sorted_ref_inds]
    cutoffs = np.diff(ref_labels).nonzero()[0] + 1
    splits = np.array_split(scores, cutoffs, axis=1)
    if aggr_mode == "max":
        aggr_scores = np.max(splits, axis=2).T
    elif aggr_mode == "min":
        aggr_scores =  np.min(splits, axis=2).T
    elif aggr_mode == "mean":
        aggr_scores =  np.mean(splits, axis=2).T
    else:
        raise f"Unknown aggr_mode: {aggr_mode}"
    if return_ref_labels:
        return aggr_scores, torch.unique(ref_labels)
    return aggr_scores

def mrr(sorted_hits, hits_cumsum=None):
    if hits_cumsum is None:
        hits_cumsum = sorted_hits.cumsum(1)
    top_ranks = np.argwhere((hits_cumsum == 1) & sorted_hits)[:, 1] + 1 # add 1 to adjust for 0 indexing
    return np.reciprocal(top_ranks.astype(np.float64)).mean().item()

def topk_acc(true_labels, scores, class_names=None, k_vals=[1]):
    true_labels = true_labels if type(true_labels) == np.ndarray else true_labels.cpu().data.numpy()
    scores = scores if type(scores) == np.ndarray else scores.cpu().data.numpy()
    if class_names is None:
        class_names = np.arange(scores.shape[1])
    else:
        class_names = class_names if type(class_names) == np.ndarray else class_names.cpu().data.numpy()
    hits = np.equal.outer(true_labels, class_names)
    sorted_socre_inds = (-1*scores).argsort(1)
    sorted_hits = np.take_along_axis(hits, sorted_socre_inds, 1)
    seen = (sorted_hits.cumsum(axis=1) > 0)
    return seen[:,np.array(k_vals)-1].mean(axis=0)

class EmbeddingEvaluator:
    def __init__(self, n_classes, shift_labels=False, ref_aggr_mode="max", pair_aggr_mode="max", score_type="euclidean", simulate_pairs=False):
        self.n_classes = n_classes
        self.shift_labels = shift_labels
        self.ref_aggr_mode = ref_aggr_mode
        self.pair_aggr_mode = pair_aggr_mode
        self.score_type = score_type
        self.simulate_pairs = simulate_pairs

    def get_metrics(self, hits, scores, k_vals=[1]):
        hits = hits if type(hits) == np.ndarray else hits.cpu().data.numpy()
        scores = scores if type(scores) == np.ndarray else scores.cpu().data.numpy()
        _,m = hits.shape
        sorted_socre_inds = (-1*scores).argsort(1)
        # sorted_scores = np.take_along_axis(scores, sorted_socre_inds, 1)
        sorted_hits = np.take_along_axis(hits, sorted_socre_inds, 1)
        hits_cumsum = sorted_hits.cumsum(axis=1)
        precisions_k = (hits_cumsum * sorted_hits)/(np.arange(m)+1)
        n_hits = sorted_hits.sum(1)
        metrics = {}
        for k in k_vals:
            metrics[f"acc_{k}"] = (hits_cumsum[:,k-1] > 0).mean().item()
            apk = precisions_k[:,:k].sum(1)/n_hits
            metrics[f"map_{k}"] = apk.mean().item()
        metrics["micro_ap"] = average_precision_score(hits, scores, average='micro')
        metrics["mrr"] = mrr(sorted_hits, hits_cumsum)
        return metrics

    def get_scores(self, query_embeddings, ref_embeddings):
        query_embeddings = query_embeddings if type(query_embeddings) == np.ndarray else query_embeddings.cpu().data.numpy()
        ref_embeddings = ref_embeddings if type(ref_embeddings) == np.ndarray else ref_embeddings.cpu().data.numpy()
        if self.score_type == "cosine_similarity":
            return cosine_similarity(query_embeddings, ref_embeddings)
        else:
            return -pairwise_distances(query_embeddings, ref_embeddings, metric=self.score_type)

    def eval(self, embeddings, labels, is_front, is_ref, single_side_eval=False, k_vals=[1, 5]):
        if self.simulate_pairs:
            pairs = create_same_side_pairs(labels[~is_ref], is_front[~is_ref])
        
        if self.shift_labels:
            labels = labels + (~is_front * self.n_classes)
        ref_labels = labels[is_ref]
        query_labels = labels[~is_ref]
        scores = self.get_scores(embeddings[~is_ref], embeddings[is_ref])

        if self.ref_aggr_mode is not None:
            scores, ref_labels = aggr_ref_axis(scores, ref_labels.cpu().data, self.ref_aggr_mode, False, True)
            ref_labels = ref_labels.to(labels.device)
        else:
            sorted_ref_inds = ref_labels.argsort()
            ref_labels = ref_labels[sorted_ref_inds]
            scores = scores[:,sorted_ref_inds]
            
        hits = (query_labels.reshape(-1,1) == ref_labels)

        if self.simulate_pairs:
            f_scores = scores[pairs[:,0],:]
            b_scores = scores[pairs[:,1],:]
            f_hits = hits[pairs[:,0],:]
            b_hits = hits[pairs[:,1],:]
            results = {}
            if single_side_eval:
                #TODO: is this ok? It feels like it is over-representing samples
                results.update({f"f_{k}":v for k,v in self.get_metrics(f_hits, f_scores, k_vals).items()})
                results.update({f"b_{k}":v for k,v in self.get_metrics(b_hits, b_scores, k_vals).items()})
                results.update({f"f_b_{k}":v for k,v in self.get_metrics(torch.concat((f_hits, b_hits)), torch.concat((f_scores, b_scores)), k_vals).items()})
            scores = side_aggregation(f_scores, b_scores, self.pair_aggr_mode)
            hits = (f_hits | b_hits)
            results.update(self.get_metrics(hits, scores, k_vals))
            return results
        else:
            return self.get_metrics(hits, scores, k_vals)

class LogitsEvaluator:
    def __init__(self, n_classes, simulate_pairs=False, shift_labels=False, side_aggr_mode="max", pair_aggr_mode="max"):
        self.n_classes = n_classes
        self.simulate_pairs = simulate_pairs
        self.shift_labels = shift_labels
        self.side_aggr_mode = side_aggr_mode
        self.pair_aggr_mode = pair_aggr_mode

    def get_metrics(self, hits, scores, k_vals=[1]):
            hits = hits if type(hits) == np.ndarray else hits.cpu().data.numpy()
            scores = scores if type(scores) == np.ndarray else scores.cpu().data.numpy()
            _,m = hits.shape
            sorted_socre_inds = (-1*scores).argsort(1)
            # sorted_scores = np.take_along_axis(scores, sorted_socre_inds, 1)
            sorted_hits = np.take_along_axis(hits, sorted_socre_inds, 1)
            hits_cumsum = sorted_hits.cumsum(axis=1)
            # precisions_k = (hits_cumsum * sorted_hits)/(np.arange(m)+1)
            # n_hits = sorted_hits.sum(1)
            metrics = {}
            for k in k_vals:
                metrics[f"acc_{k}"] = (hits_cumsum[:,k-1] > 0).mean().item()
                # apk = precisions_k[:,:k].sum(1)/n_hits
                # metrics[f"map_{k}"] = apk.mean().item()
            metrics["micro_ap"] = average_precision_score(hits, scores, average='micro')
            metrics["mrr"] = mrr(sorted_hits, hits_cumsum)
            return metrics
    
    def eval(self, logits, labels, is_front, is_ref=None, single_side_eval=False, exclude_refs=False, k_vals=[1,5]):
        _,n = logits.shape
        if n > self.n_classes:
            assert (2*self.n_classes == n)
            if self.shift_labels:
                labels = torch.where(is_front, labels, labels + self.n_classes)
            else:
                n = self.n_classes
                f_logits = logits[:, :n]
                b_logits = logits[:, n:]
                if self.side_aggr_mode is not None:
                    logits = side_aggregation(f_logits, b_logits, self.side_aggr_mode)
                else:
                    logits = torch.where(is_front.reshape(-1,1), f_logits, b_logits)
        unique_classes = torch.arange(n).to(logits.device)
        labels = labels[~is_ref] if is_ref is not None and exclude_refs else labels

        hits = (labels.reshape(-1,1) == unique_classes)

        if self.simulate_pairs:
            pairs = create_same_side_pairs(labels, is_front, is_ref)
            f_logits = logits[pairs[:,0],:]
            f_hits = hits[pairs[:,0], :]
            b_logits = logits[pairs[:,1],:]
            b_hits = hits[pairs[:,1], :]
            results = {}
            if single_side_eval:
                #TODO: is this ok? It feels like it is over-representing samples
                results.update({f"f_{k}":v for k,v in self.get_metrics(f_hits, f_logits, k_vals).items()})
                results.update({f"b_{k}":v for k,v in self.get_metrics(b_hits, b_logits, k_vals).items()})
                results.update({f"f_b_{k}":v for k,v in self.get_metrics(torch.concat((f_hits, b_hits)), torch.concat((f_logits, b_logits)), k_vals).items()})
            logits = side_aggregation(f_logits, b_logits, self.pair_aggr_mode)
            assert f_hits == b_hits
            hits = f_hits
            results.update(self.get_metrics(hits, logits, k_vals))
            return results
        else:
            return self.get_metrics(hits, logits, k_vals)

class Embedder:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

    def embed_all(self, include_side_labels=True, include_ref_labels=True):
        outputs = {"labels": []}
        if include_side_labels:
            outputs["is_front"] = []
        if include_ref_labels:
            outputs["is_ref"] = []
        self.model.eval()
        for data in tqdm(self.dataloader, total=len(self.dataloader)):
            imgs = data["img"].to(self.device)
            labels = data["label"].long().to(self.device)
            outputs["labels"].append(labels)
            if include_side_labels:
                outputs["is_front"].append(data["is_front"].bool().to(self.device))
            if include_ref_labels:
                outputs["is_ref"].append(data["is_ref"].bool().to(self.device))
            with torch.no_grad():
                model_outputs = self.model(imgs, labels)
                for k,v in model_outputs.items():
                    if k not in outputs:
                        outputs[k] = [v]
                    else:
                        outputs[k].append(v)
        for k,v in outputs.items():
            outputs[k] = torch.cat(v, 0)
        return outputs
    
