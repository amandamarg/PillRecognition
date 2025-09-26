import numpy as np

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