from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score
import numpy as np 
import torch
import warnings

class Metrics:
    def __init__(self, history=[], aggregation_mode=None, per_class=False, n_classes=None):
        assert aggregation_mode in ["min", "max", "sum", "avg", None]
        self.aggregation_mode = aggregation_mode
        self.history = history.copy()
        self.per_class = per_class
        if per_class:
            assert n_classes is not None
            self.n_classes = n_classes
        else:
            self.n_classes = None

    def calculate(self, hits, labels=None, update_history=True):
        pass

    def aggregate(self, metric_vals, counts=None):
        if self.aggregation_mode == "min":
            return metric_vals.nanmin(axis=int(self.per_class))
        elif self.aggregation_mode == "max":
            return metric_vals.nanmax(axis=int(self.per_class))
        elif self.aggregation_mode == "sum":
            return metric_vals.nansum(axis=int(self.per_class))
        elif self.aggregation_mode == "avg":
            if counts is None:
                counts = len(metric_vals)
            counts = counts - (metric_vals.isnan()).sum(axis=int(self.per_class))
            return metric_vals.nansum(axis=int(self.per_class))/counts
        else:
            return metric_vals
        
    def get_best_val(self, return_index=True):
        best_val = self.history.min(axis=int(self.per_class))
        if return_index:
            best_idx = self.history.argmin(axis=int(self.per_class))
            return best_val, best_idx
        return best_val

    
class AP_K(Metrics):
    def __init__(self, k, replace_nan=None, use_min=False, history=[], aggregation_mode=None, per_class=False, n_classes=None):
        super().__init__(history=history, aggregation_mode=aggregation_mode, per_class=per_class, n_classes=n_classes)
        self.k = k
        self.use_min = use_min #if True, then for an input where cols = # columns, will use the min(cols,k) for k, otherwise if cols < k, will return np.nan
        self.replace_nan = replace_nan

    def calculate(self, hits, labels=None, update_history=True):
        _,cols = hits.shape
        k = min(cols, self.k)
        if cols < self.k:
            warnings.warn("{:d} cols is less than k={:d}".format(cols,self.k))
            if not self.use_min:
                return np.nan
        
        top_k = hits[:,:k]
        precisions = (top_k.cumsum(axis=1) * top_k)/np.arange(1,k+1)
        summed_precisions = precisions.sum(axis=1)
        N = hits.sum(axis=1)

        replace_val = self.replace_nan if self.replace_nan is not None else np.nan
        metric_val = np.where(N > 0, summed_precisions/N, replace_val)

        if self.per_class:
            grouped_labels = np.equal.outer(np.arange(self.n_classes), labels)
            metric_vals = grouped_labels * metric_vals
            counts = grouped_labels.sum(axis=1)
        else:
            counts = hits.sum()
        
        metric_val = self.aggregate(metric_vals, counts)
        if update_history:
            self.history.append(metric_val)
        return metric_val

class MAP_K(AP_K):
    def __init__(self, k, replace_nan=None, use_min=False, history=[], per_class=False, n_classes=None):
        super().__init__(k=k, replace_nan=replace_nan, use_min=use_min, history=history, aggregation_mode="avg", per_class=per_class, n_classes=n_classes)

class MRR(Metrics):
    def __init__(self, history=[], per_class=False, n_classes=None):
        super().__init__(history=history, aggregation_mode="avg", per_class=per_class, n_classes=n_classes)

    def calculate(self, hits, labels=None, update_history=True):
        hits = hits & (np.cumsum(hits, axis=1) == 1) #this is to make sure we are getting the first hit in each row
        inds, top_rank = np.argwhere(hits)
        top_rank = top_rank + 1 #shift ranks so start at 1 instead of 0
        replace_val = self.replace_nan if self.replace_nan is not None else np.nan
        reciprocal_ranks = np.where(top_rank > 0, 1/top_rank, replace_val)
        if self.per_class:
            grouped_labels = np.equal.outer(np.arange(self.n_classes), labels)
            metric_vals = grouped_labels * metric_vals
            counts = grouped_labels.sum(axis=1)
        else:
            counts = hits.sum()
        metric_val = self.aggregate(reciprocal_ranks, counts)
        if update_history:
            self.history.append(metric_val)
        return metric_val
    
class MetricTracker:
    def __init__(self, logit_metrics, embedding_metrics, use_refs=False, split_embeddings=False, shift_back_labels=False, n_classes=None):
        self.logit_metrics = logit_metrics.deepcopy()
        self.embedding_metrics = embedding_metrics.deepcopy()
        self.batch_embeddings = []
        self.batch_logits = []
        self.batch_labels = []
        self.refs = [] if use_refs else None
        self.split_embeddings = split_embeddings
        self.shift_back_labels = shift_back_labels
        self.front = None
        if self.shift_back_labels and not self.split_embeddings:
            self.front = []
        self.n_classes = n_classes

    def update_batch(self, embeddings, logits, labels, refs=None, front=None):
        self.batch_embeddings.extend(embeddings.clone().detach())
        self.batch_logits.extend(logits.clone().detach())
        self.batch_labels.extend(labels.clone().detach())
        if self.refs is not None and refs is not None:
            self.refs.extend(refs.clone().detach())
        if self.front is not None and front is not None:
            self.front.extend(front.clone().detach())
    
    def clear_batch(self):
        self.batch_embeddings = []
        self.batch_logits = []
        self.batch_labels = []
        if self.refs is not None:
            self.refs = []
        if self.front is not None:
            self.front = []

    def embedding_hits(self):
        batch_embeddings = np.array(self.batch_embeddings)
        batch_labels = np.array(self.batch_labels)
        if self.split_embeddings:
            batch_embeddings = np.vstack(batch_embeddings.hsplit(2))
            batch_labels = np.hstack((batch_labels, self.n_classes + batch_labels)) if self.shift_back_labels else np.hstack((batch_labels, batch_labels))
        elif self.shift_back_labels:
            batch_labels = np.where(batch_labels, batch_labels, batch_labels + self.n_classes)

        if self.refs is not None:
            refs = np.array(self.refs)
            if self.split_embeddings:
                refs = np.hstack((refs, refs))
            ref_labels = batch_labels[refs]
            ref_embeddings = batch_embeddings[refs]
            cons_labels = batch_labels[~refs]
            cons_embeddings = batch_embeddings[~refs]
            distance_matrix = euclidean_distances(cons_embeddings, ref_embeddings)
            sorted_rankings = distance_matrix.argsort(axis=1)
            same_labels = np.argwhere(np.equal.outer(cons_labels, ref_labels))
            same_labels[:,1] = np.argwhere(sorted_rankings[:,0] == same_labels[:,1].reshape(-1,1)[:,1])[:,1]
        else:
            distance_matrix = euclidean_distances(batch_embeddings, batch_embeddings)
            distances = distances - np.identity(len(distances)) #this is done to make sure that distances between same images are pushed to front when sorting, making them easy to exlude
            sorted_rankings = distances.argsort(axis=1)
            sorted_rankings = sorted_rankings[:,1:] #exclude same images
            same_labels = np.argwhere(np.equal.outer(batch_labels, batch_labels))
            same_labels = same_labels[same_labels[:,0] != same_labels[:,1]] #exclude same images

        hits = np.zeros(sorted_rankings.shape)
        hits[same_labels[:,0], same_labels[:,0]] = 1
        return hits
    
    def logit_hits(self):
        sorted_ranks = np.argsort(np.array(self.batch_logits), axis=1)
        return (sorted_ranks == np.array(self.batch_labels).reshape(-1,1))

    def update_metrics(self):
        embedding_hits = self.embedding_hits()
        for metric in self.embedding_metrics.values():
            metric.calculate(embedding_hits, np.array(self.batch_labels), True)
        
        logit_hits = self.logit_hits()
        for metric in self.logit_metrics.values():
            metric.calculate(logit_hits, np.array(self.batch_labels), True)

        self.clear_batch()



