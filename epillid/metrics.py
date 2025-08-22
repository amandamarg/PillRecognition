from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score
import numpy as np 
import torch
import warnings

class Metrics:
    def __init__(self, best_is):
        assert best_is in ["max", "min", 0]
        self.best_is = best_is

    def calculate(self, hits):
        pass
    
class AP_K(Metrics):
    def __init__(self, k, replace_nan=None, use_min=False):
        super().__init__(best_is="max")
        self.k = k
        self.use_min = use_min #if True, then for an input where cols = # columns, will use the min(cols,k) for k, otherwise if cols < k, will return None

    def calculate(self, hits):
        _,cols = hits.shape
        k = min(cols, self.k)
        if cols < self.k:
            warnings.warn("{:d} cols is less than k={:d}".format(cols,self.k))
            if not self.use_min:
                return None
        
        top_k = hits[:,:k]
        precisions = (top_k.cumsum(axis=1) * top_k)/np.arange(1,k+1)
        summed_precisions = precisions.sum(axis=1)
        N = hits.sum(axis=1)

        replace_val = self.replace_nan if self.replace_nan is not None else np.nan

        return np.where(N > 0, summed_precisions/N, replace_val)


class MAP_K(AP_K):
    def __init__(self, k, drop_nan=True, use_min=False, per_class=False):
        replace_nan = None if drop_nan else 0
        super().__init__(k=k,replace_nan=replace_nan,use_min=use_min)
        self.per_class = per_class
        self.drop_nan = drop_nan

    def calculate(self, hits, labels=None):
        apk = super().calculate(hits=hits)
        if apk is None:
            return None
        if self.drop_nan:
            if labels is not None:
                labels = labels[np.isnan(apk)]
            apk = apk[np.isnan(apk)]
        if not self.per_class:
            return apk.sum()/len(apk)
        unique_labels = labels.unique()
        grouped_labels = np.equal.outer(unique_labels, labels)
        counts = grouped_labels.sum(axis=1)
        assert (counts > 0).all()
        mapk = (grouped_labels @ apk)/counts
        return (unique_labels, mapk)
    

class MRR(Metrics):
    def __init__(self, drop_nan=True, per_class=False):
        super().__init__(best_is="max")
        self.drop_nan = drop_nan
        self.per_class = per_class

    def calculate(self, hits, labels=None):
        hits = hits & (np.cumsum(hits, axis=1) == 1) #this is to make sure we are getting the first hit in each row
        inds, top_rank = np.argwhere(hits)
        top_rank = top_rank + 1 #shift ranks so start at 1 instead of 0

        if not self.per_class:
            n = len(top_rank) if self.drop_nan else len(hits)
            return top_rank.sum()/n
        if self.drop_nan:
            labels = labels[inds]
            reciprocal_rank = 1/top_rank
        else:
            reciprocal_rank = np.zeros(len(labels))
            reciprocal_rank[inds] = (1/top_rank)
        unique_labels = labels.unique()
        grouped_labels = np.equal.outer(unique_labels, labels)
        sum_rr = (grouped_labels @ reciprocal_rank)
        counts = grouped_labels.sum(axis=1)
        assert (counts > 0).all()
        return (unique_labels, sum_rr/counts)
    