from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.metrics import top_k_accuracy_score
import numpy as np 

def get_classification_accuracy(true_labels, logits):
    assert true_labels.device.type=='cpu' and logits.device.type=='cpu'
    all_labels = np.arange(start=0,stop=logits.shape[1])
    top_1_accuracy = top_k_accuracy_score(true_labels, logits, k=1, labels=all_labels)
    top_5_accuracy = top_k_accuracy_score(true_labels, logits, k=5, labels=all_labels)
    return top_1_accuracy, top_5_accuracy

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
    assert m >= k

    sum_prec_k = np.zeros(n)
    for i in range(1,k+1):
        sum_prec_k += ((hits[:,:i].sum(axis=1)/i)*hits[:,i-1])
    
    N = hits.sum(axis=1)

    return sum_prec_k/np.where(N > 0, N, np.nan)

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

def MRR(hits, replace_nan=None, per_class=False, labels=None):
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
