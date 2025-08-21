from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score
import numpy as np 
import torch
import warnings

class Metrics:
    def __init__(self, best_is, replace_nan=None):
        assert best_is in ["max", "min", 0]
        self.best_is = best_is
        self.replace_nan = replace_nan

    def calculate(self, inputs):
        pass
    
class AP_K(Metrics):
    def __init__(self, k, replace_nan=None, use_min=False):
        super().__init__(best_is="max", replace_nan=replace_nan)
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

        return sum_prec_k/np.where(N > 0, N, replace_val)


class MAP_K(AP_K):
    def __init__(self, k, replace_nan=None, use_min=False, per_class=False):
        super().__init__(k=k,replace_nan=replace_nan,use_min=use_min)
        self.per_class = per_class

    def calculate(self, inputs, labels=None):
        apk = super().calculate(inputs=inputs)
        if self.per_class:
            assert labels is not None
            
        return
