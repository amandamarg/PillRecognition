import torch
import torch.nn as nn
import torch.nn.functional as F

def get_dist_matrix(embeddings, p=2.0):
    n = embeddings.shape[0]
    inds = torch.arange(n).to(embeddings.device)
    i,j = torch.meshgrid(inds, inds)
    idx_mat = torch.where(i<j, (n-1)*i-(i-1)*i/2 + (j-i-1), (n-1)*j-(j-1)*j/2 + (i-j-1)).long()
    distances = F.pdist(embeddings, p=p)
    dist_matrix = torch.where(idx_mat >= 0, distances[idx_mat], 0.0).fill_diagonal_(0.0)
    return dist_matrix

def get_cosine_dist_matrix(embeddings):
    n = embeddings.shape[0]
    inds = torch.arange(n).to(embeddings.device)
    i,j = torch.meshgrid(inds, inds)
    all_pairs = torch.column_stack((i.reshape(-1), j.reshape(-1)))
    distances = 1 - F.cosine_similarity(embeddings[all_pairs[:,0]], embeddings[all_pairs[:,1]])
    dist_matrix = distances.reshape(n,n)
    return dist_matrix

def get_dist_matrix_x_y(x_embeddings, y_embeddings, use_cosine_distance=False):
    n, m = x_embeddings.shape[0], y_embeddings.shape[0]
    x_inds = torch.arange(n).to(x_embeddings.device)
    y_inds = torch.arange(m).to(y_embeddings.device)
    i,j = torch.meshgrid(x_inds, y_inds)
    all_pairs = torch.column_stack((i.reshape(-1), j.reshape(-1)))
    if use_cosine_distance:
        distances = 1 - F.cosine_similarity(x_embeddings[all_pairs[:,0]], y_embeddings[all_pairs[:,1]])
    else:
        distances = F.pairwise_distance(x_embeddings[all_pairs[:,0]], y_embeddings[all_pairs[:,1]])
    dist_matrix = distances.reshape(n,m)
    return dist_matrix

def get_pairs(labels, is_front=None, is_ref=None):
    n = labels.shape[0]
    inds = torch.arange(n).to(labels.device)
    i,j = torch.meshgrid(inds, inds)
    all_pairs = torch.column_stack((i.reshape(-1), j.reshape(-1)))
    all_pairs = all_pairs[all_pairs[:,0] != all_pairs[:,1]] #exclude indicies paired with self
    if is_front is not None:
        all_pairs = all_pairs[is_front[all_pairs[:,0]] == is_front[all_pairs[:,1]]] #only include pairs that are same side
    if is_ref is not None:
        all_pairs = all_pairs[~is_ref[all_pairs[:,0]] & is_ref[all_pairs[:,1]]] #exclude pairs where first elements is cons and second element is ref
    labeled_pairs = labels[all_pairs]
    same_labels = (labeled_pairs[:,0] == labeled_pairs[:,1])
    pos_pairs = all_pairs[same_labels]
    neg_pairs = all_pairs[~same_labels]
    return pos_pairs, neg_pairs

def get_triplets(labels, is_front=None, is_ref=None):
    pos_pairs, neg_pairs = get_pairs(labels, is_ref, is_front)
    same_anchors = torch.argwhere((pos_pairs[:,0].reshape(-1,1) == neg_pairs[:,0]))
    triplets = torch.column_stack((pos_pairs[same_anchors[:,0]], neg_pairs[same_anchors[:,1], 1]))
    return triplets

def get_hardest_triplets(embeddings, labels, is_front=None, is_ref=None, use_cosine_dist=False):
    if use_cosine_dist=='cosine':
        dist_mat = get_cosine_dist_matrix(embeddings)
    else:
        dist_mat = get_cosine_dist_matrix(embeddings)
    pos_mask = (labels.reshape(-1,1) == labels)
    neg_mask = ~pos_mask
    if is_front is not None:
        pos_mask &= (is_front.reshape(-1,1) == is_front)
        neg_mask &= (is_front.reshape(-1,1) == is_front)
    if is_ref is not None:
        pos_mask &= is_ref
        neg_mask &= is_ref

    hardest_pos_dist, hardest_pos_inds = torch.where(pos_mask, dist_mat, -torch.inf).fill_diagonal_(-torch.inf).max(dim=1)
    hardest_neg_dist, hardest_neg_inds = torch.where(neg_mask, dist_mat, torch.inf).min(dim=1)
    anchors = torch.arange(embeddings.shape[0]).to(embeddings.device)
    hardest_triplets = torch.column_stack((anchors, hardest_pos_inds, hardest_neg_inds))
    if is_ref is not None:
        return hardest_triplets[~is_ref], hardest_pos_dist[~is_ref], hardest_neg_dist[~is_ref]
    return hardest_triplets, hardest_pos_dist, hardest_neg_dist

def triplet_selector(embeddings, triplets, margin=.2, mode='all', use_cosine_dist=False):
    assert mode in ['easy', 'hard', 'semihard', 'all', None]
    a,p,n = triplets.T
    if use_cosine_dist:
        pos_dists = 1 - F.cosine_similarity(embeddings[a], embeddings[p])
        neg_dists = 1 - F.cosine_similarity(embeddings[a], embeddings[n])
    else:
        pos_dists = F.pairwise_distance(embeddings[a], embeddings[p])
        neg_dists = F.pairwise_distance(embeddings[a], embeddings[n])

    if mode == None:
        return triplets, pos_dists, neg_dists

    min_thresh = 0 if mode == 'semihard' else margin if mode == 'easy' else -torch.inf
    max_thresh = torch.inf if mode == 'easy' else 0 if mode == 'hard' else margin
    diff = neg_dists - pos_dists
    condition_mask = (diff > min_thresh) & (diff <= max_thresh)
    return triplets[condition_mask], pos_dists[condition_mask], neg_dists[condition_mask]

class Miner(nn.Module):
    def __init__(self, margin=0.2, mode='all', use_cosine_dist=False):
        super(Miner, self).__init__()
        self.margin=margin
        self.mode=mode
        self.use_cosine_dist=use_cosine_dist

    def forward(self, embeddings, labels, is_front=None, is_ref=None):
        if self.mode == 'hardest':
            triplets, pos_dist, neg_dist = get_hardest_triplets(embeddings, labels, is_front, is_ref, self.use_cosine_dist)
        else:
            triplets = get_triplets(labels, is_front, is_ref)
            triplets, pos_dist, neg_dist = triplet_selector(embeddings, triplets, self.margin, self.mode, self.use_cosine_dist)
        return triplets, pos_dist, neg_dist

# from sklearn.metrics.pairwise import pairwise_distances
# import numpy as np
# def get_triplets(x_embeddings, x_labels, y_embeddings=None, y_labels=None, margin=.02, mode='all', distance_type='cosine'):
#     assert mode in ['easy', 'hard', 'semihard', 'all', None]

#     x_is_y = False
#     if y_embeddings is None or y_labels is None:
#         y_embeddings = x_embeddings
#         y_labels = x_labels
#         x_is_y = True


#     same_label_mask = np.equal.outer(x_labels, y_labels)
#     same_label_pairs = np.argwhere(same_label_mask)
#     if x_is_y:
#         same_label_pairs = same_label_pairs[same_label_pairs[:,0] != same_label_pairs[:,1]]

#     distances = pairwise_distances(x_embeddings, y_embeddings, metric=distance_type)

#     if mode == None:
#         diff_pairs = np.argwhere(~same_label_mask[same_label_pairs[:,0]])
#         same_label_pairs = same_label_pairs[diff_pairs[:,0]]
#         return np.c_[same_label_pairs, diff_pairs[:,1]], distances

#     same_pair_distances = distances[same_label_pairs[:,0], same_label_pairs[:,1]].reshape(-1,1)
#     diff_pair_distances = distances[same_label_pairs[:,0]]
#     diff = diff_pair_distances - same_pair_distances

#     condition_mask = (diff > margin) if mode == 'easy' else (diff <= 0) if (mode=='hard') else ((diff <= margin) & (diff > 0))
#     triplets = np.argwhere((~same_label_mask[same_label_pairs[:,0]]) & condition_mask)
#     triplets = np.c_[same_label_pairs[triplets[:,0]], triplets[:,1]]
#     return triplets, distances


# def get_triplets(embeddings, labels, is_front=None, is_ref=None, margin=.2, mode='hard', distance_type='cosine'):
#     assert mode in ['easy', 'hard', 'semihard', 'all', None]

#     same_label_mask = np.equal.outer(labels, labels)
#     pos_pairs = np.argwhere(same_label_mask)
#     pos_pairs = pos_pairs[pos_pairs[:,0] != pos_pairs[:,1]]
#     neg_pairs = np.argwhere(~same_label_mask)
#     if is_front is not None:
#         pos_pairs = pos_pairs[is_front[pos_pairs[:,0]] == is_front[pos_pairs[:,1]]]
#         neg_pairs = neg_pairs[is_front[neg_pairs[:,0]] == is_front[neg_pairs[:,1]]]
#     if is_ref is not None:
#         pos_pairs = pos_pairs[~is_ref[pos_pairs[:,0]] & is_ref[pos_pairs[:,1]]]
#         neg_pairs = neg_pairs[~is_ref[neg_pairs[:,0]] & is_ref[neg_pairs[:,1]]]
    
#     pair_idx = np.argwhere(pos_pairs[:,0] == neg_pairs[:,0].reshape(-1,1))
#     pos_pairs = pos_pairs[pair_idx[:,0]]
#     neg_pairs = neg_pairs[pair_idx[:,1]]
#     triplets =  np.c_[pos_pairs, neg_pairs[:,1]]

#     distances = pairwise_distances(embeddings, metric=distance_type)
#     pos_pair_distances = distances[pos_pairs]
#     neg_pair_distances = distances[neg_pairs]

#     if mode == None:
#         return triplets, pos_pair_distances, neg_pair_distances

#     diff = neg_pair_distances - pos_pair_distances

#     min_thresh = 0 if mode == 'semihard' else margin if mode == 'easy' else -np.inf
#     max_thresh = np.inf if mode == 'easy' else 0 if mode == 'hard' else margin
#     condition_mask = (diff > min_thresh) & (diff <= max_thresh)
    
#     triplets = triplets[condition_mask]
#     pos_pair_distances = pos_pair_distances[condition_mask]
#     neg_pair_distances = pos_pair_distances[condition_mask]
#     return triplets, pos_pair_distances, neg_pair_distances