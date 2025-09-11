from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score, average_precision_score
import numpy as np 
import torch
from utils import embed_all

def compare_labels(labels1, labels2):
    assert labels1.device.type=='cpu' and labels2.device.type=='cpu'
    return np.equal.outer(labels1.numpy(), labels2.numpy())

def get_hits(labels, embeddings, ref_labels=None, ref_embeddings=None, return_distances=False, return_sorted_rankings=False, return_same_pairs=False):
    ''' 
    If either ref_labels or ref_embeddings is missing, then will compute hits between sambles in same batch, otherwise will compute hits between the first set of embeddings and the second set
    If computing between two different sets of embeddings, it is assumed that there is no overlap between the samples
    '''

    assert labels.device.type=='cpu' and embeddings.device.type=='cpu'
    outputs = []
    if ref_labels is not None and ref_embeddings is not None:
        assert ref_labels.device.type=='cpu' and ref_embeddings.device.type=='cpu'
        distances = euclidean_distances(embeddings, ref_embeddings)
        sorted_rankings = distances.argsort(axis=1)
        same_labels = compare_labels(labels, ref_labels)
        same_pairs = np.argwhere(same_labels)

    else:
        distances = euclidean_distances(embeddings, embeddings)
        distances = distances - np.identity(len(distances)) #this is done to make sure that distances between same images are pushed to front when sorting, making them easy to exlude
        sorted_rankings = distances.argsort(axis=1)
        sorted_rankings = sorted_rankings[:,1:] #exclude same images

        same_labels = compare_labels(labels, labels)
        same_pairs = np.argwhere(same_labels)
        same_pairs = same_pairs[same_pairs[:,0] != same_pairs[:,1]] #exclude same images


    true_ranks = np.stack([same_pairs[:,0], np.argwhere(sorted_rankings[same_pairs[:,0]] == same_pairs[:,1].reshape(-1,1))[:,1]], axis=1)
    n,m = sorted_rankings.shape
    hits = np.zeros((n, m))
    hits[true_ranks[:,0], true_ranks[:,1]] = 1
    outputs.append(hits)
    if return_distances:
        outputs.append(distances)
    if return_sorted_rankings:
        outputs.append(sorted_rankings)
    if return_same_pairs:
        outputs.append(same_pairs)
    return tuple(outputs)

def ap_k(hits, k):
    ''' 
    Calculates the average precision at k
    '''
    n,m = hits.shape

    if (m < k):
        print("Warning: Sample too small for k=" + str(k))
        print("Using k=" + str(m) + " instead")
        print(hits.shape)
        k = m


    top_k = hits[:,:k]
    precisions = (top_k.cumsum(axis=1) * top_k)/np.arange(1,k+1)
    summed_precisions = precisions.sum(axis=1)
    N = hits.sum(axis=1)


    return np.where(N > 0, summed_precisions/N, np.nan)

def map_k(hits, k, replace_nan=None, per_class=False, labels=None):
    ''' 
    Calculates the mean average precision at k
    if replace_nan is None, then will default to dropping nan values before calculating mean, otherwise will replace nans with whatever is input into replace_nan
    if per_class is true, labels is a required argument
    '''
    ap = ap_k(hits, k)
    if not per_class:
        ap = ap[~np.isnan(ap)] if (replace_nan is None) else np.nan_to_num(ap, nan=replace_nan)
        return np.mean(ap)

    assert labels.device.type == 'cpu'
    group_labels = np.triu(compare_labels(labels, labels))
    if replace_nan is not None:
        ap = np.nan_to_num(ap, nan=replace_nan)
        return (group_labels @ ap)/group_labels.sum(axis=1)
    else:
        map_per_class = (group_labels @ np.nan_to_num(ap, 0.0))/group_labels.sum(axis=1)
        return np.where(np.isnan(ap), np.nan, map_per_class)

def get_top_rank(hits, shift_ranks_1=False):
    top_ranks = np.argmax(hits, axis=1)
    top_ranks = np.where(np.sum(hits, axis=1) == 0, np.nan, top_ranks)
    if shift_ranks_1:
        return top_ranks + 1
    return top_ranks

def MRR_embed(hits, replace_nan=None, per_class=False, labels=None):
    '''default behavior is to propogate nans when calculating per_class or exclude nans from calculations otherwise'''
    top_ranks = get_top_rank(hits, True)
    if per_class:
        assert labels.device.type == 'cpu'
        group_labels = np.triu(compare_labels(labels, labels))
        reciprocal_ranks = group_labels @ np.where(np.isnan(top_ranks), 0.0, 1/top_ranks)
        if replace_nan is None:
            return np.where(np.isnan(top_ranks), np.nan, np.sum(reciprocal_ranks, axis=1)/np.sum(group_labels, axis=1))
        else:
            return np.where(np.isnan(top_ranks), replace_nan, np.sum(reciprocal_ranks, axis=1)/np.sum(group_labels, axis=1))
    if replace_nan is None:
        return np.mean((1/top_ranks[~np.isnan(top_ranks)]))
    return np.mean(np.where(np.isnan(top_ranks), replace_nan, 1/top_ranks))

def MRR(true_labels, logits, per_class=False):
    assert true_labels.device.type=='cpu' and logits.device.type=='cpu'
    batch_size, n_classes = logits.shape
    ranked_predictions = torch.argsort(logits, dim=1, descending=True)
    ranking_mask = (torch.argsort(logits,dim=1, descending=True) == true_labels.reshape(len(true_labels), -1))
    rank_of_true_label = torch.argwhere(ranking_mask)
    reciprocal_rank = (1/rank_of_true_label[:,1])
    if per_class:
        unique_labels, inv_unique_labels, counts_labels = torch.unique(true_labels, return_inverse=True, return_counts=True)
        ordered_reciprocal_rankings = torch.zeros((len(unique_labels), batch_size), dtype=torch.float)
        ordered_reciprocal_rankings[inv_unique_labels, rank_of_true_label[:,0]] = reciprocal_rank
        return (unique_labels, ordered_reciprocal_rankings.sum(dim=1)/counts_labels)
    return reciprocal_rank.sum()/batch_size


def eval_embeddings(labels, embeddings, ref_labels=None, ref_embeddings=None):
    assert labels.device.type=='cpu' and embeddings.device.type=='cpu'

    if ref_labels is None:
        hits, = get_hits(labels, embeddings)
    else:
        assert (ref_embeddings is not None)
        assert ref_labels.device.type=='cpu' and ref_embeddings.device.type=='cpu'

        hits, = get_hits(labels, embeddings, ref_labels, ref_embeddings)

    return {'map_1': map_k(hits, 1), 'map_5': map_k(hits, 5), 'MRR': MRR_embed(hits), }

def eval_logits(labels, logits):
    assert labels.device.type=='cpu' and logits.device.type=='cpu'
    all_labels = np.arange(start=0,stop=logits.shape[1])
    return {'top_1_accuracy': top_k_accuracy_score(labels, logits, k=1, labels=all_labels), 'top_5_accuracy':top_k_accuracy_score(labels, logits, k=5, labels=all_labels), 'MRR': MRR(labels, logits)}

import torch.nn.functional as F
from sklearn.metrics import label_ranking_average_precision_score

def get_similarity_scores(embeddings, labels):
    n = embeddings.shape[0]
    inds = torch.arange(n).to(embeddings.device)
    i,j = torch.meshgrid(inds, inds)
    all_pairs = torch.column_stack((i.reshape(-1), j.reshape(-1)))
    all_pairs = all_pairs[all_pairs[:,0] != all_pairs[:,1]]
    scores = F.cosine_similarity(embeddings[all_pairs[:,0]], embeddings[all_pairs[:,1]]).reshape(n,n-1)
    shifted_labels = labels[torch.where(i<=j, j+1, j)[:,0:-1]]
    true_label_matrix = (shifted_labels == labels[i[:,0:-1]])
    return scores, true_label_matrix

def get_similarity_scores_x_y(x_embeddings, x_labels, y_embeddings, y_labels):
    n, m = x_embeddings.shape[0], y_embeddings.shape[0]
    x_inds = torch.arange(n).to(x_embeddings.device)
    y_inds = torch.arange(m).to(y_embeddings.device)
    i,j = torch.meshgrid(x_inds, y_inds)
    all_pairs = torch.column_stack((i.reshape(-1), j.reshape(-1)))
    scores = F.cosine_similarity(x_embeddings[all_pairs[:,0]], y_embeddings[all_pairs[:,1]]).reshape(n,m)
    true_label_matrix = (x_labels.reshape(-1,1) == y_labels)
    return scores, true_label_matrix

def eval_metrics(inputs, n_classes):
    embeddings = inputs['embeddings']
    logits = inputs['logits']
    labels = inputs['labels']
    is_front = inputs.get('is_front', None)
    is_ref = inputs.get('is_ref', None)

    unique_classes = np.arange(n_classes)
    top_1_acc = top_k_accuracy_score(labels, logits, k=1, labels=unique_classes)
    top_5_acc = top_k_accuracy_score(labels, logits, k=5, labels=unique_classes)

    if is_ref is not None:
        ref_embeddings = embeddings[is_ref]
        cons_embeddings = embeddings[~is_ref]
        ref_labels = labels[is_ref]
        cons_labels = labels[~is_ref]
        scores, true_label_matrix = get_similarity_scores_x_y(cons_embeddings, cons_labels, ref_embeddings, ref_labels)        
    else:
        scores, true_label_matrix = get_similarity_scores(embeddings, labels)

    lraps = label_ranking_average_precision_score(y_true=true_label_matrix, y_score=scores)

    metric_vals = {'top_1_acc': top_1_acc, 'top_5_acc': top_5_acc, "LRAPS": lraps}
    return metric_vals
class MetricTracker:
    def __init__(self):
        self.history = {}
        self.curr_metrics = {}

    def update_curr_metrics(self, inputs):
        for k,v in inputs.items():
            if k not in self.curr_metrics.keys():
                self.curr_metrics[k] = torch.tensor([v])
            else:
                self.curr_metrics[k] = torch.concat((self.curr_metrics[k], torch.tensor([v])))
    
    def update_history(self):
        outputs = {}
        for k,v in self.curr_metrics.items():
            outputs[k] = torch.mean(v)
            if k not in self.history.keys():
                self.history[k] = outputs[k].ravel()
            else:
                self.history[k] = torch.concat((self.history[k], outputs[k].ravel()))
            self.curr_metrics[k] = torch.tensor([])
        return outputs