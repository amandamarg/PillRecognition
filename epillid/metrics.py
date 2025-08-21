from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score
import numpy as np 
import torch
import warnings

class Metrics:
    def __init__(self, best_is):
        assert best_is in ["max", "min", 0]
        self.best_is = best_is

    def calculate(self, inputs):
        pass
    
class AP_K(Metrics):
    def __init__(self, k, replace_nan=None, use_min=False):
        super().__init__(best_is="max")
        self.k = k
        self.use_min = use_min #if True, then for an input where cols = # columns, will use the min(cols,k) for k, otherwise if cols < k, will return None

    def calculate(self, inputs):
        _,cols = inputs.shape
        k = min(cols, self.k)
        if cols < self.k:
            warnings.warn("{:d} cols is less than k={:d}".format(cols,self.k))
            if not self.use_min:
                return None
        
        sum_prec_k = np.zeros(cols)
        for i in range(1,k+1):
            sum_prec_k += ((inputs[:,:i].sum(axis=1)/i)*inputs[:,i-1])
    
        N = inputs.sum(axis=1)

        replace_val = self.replace_nan if self.replace_nan is not None else np.nan

        return np.where(N > 0, sum_prec_k/N, replace_val)


class MAP_K(AP_K):
    def __init__(self, k, drop_nan=True, use_min=False, per_class=False):
        replace_nan = None if drop_nan else 0
        super().__init__(k=k,replace_nan=replace_nan,use_min=use_min)
        self.per_class = per_class
        self.drop_nan = drop_nan

    def calculate(self, inputs, labels=None):
        apk = super().calculate(inputs=inputs)
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

