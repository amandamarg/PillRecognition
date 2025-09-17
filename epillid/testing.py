from sklearn.metrics import top_k_accuracy_score, average_precision_score, label_ranking_average_precision_score, pairwise_distances
import numpy as np 
import torch
from tqdm import tqdm

def embed_all(model, dataloader, use_side_labels=True, use_ref_labels=True):
    device = model.device
    outputs = {"labels": [], "embeddings": [], "logits": []}
    if use_side_labels:
        outputs["is_front"] = []
    if use_ref_labels:
        outputs["is_ref"] = []

    for data in tqdm(dataloader, total=len(dataloader)):
        outputs["label"].append(data["label"].long().to(device))
        imgs = data["img"].to(device)
        if use_side_labels:
            outputs["is_front"].append(data["is_front"].to(device))
        if use_ref_labels:
            outputs["is_ref"].append(data["is_ref"].to(device))
        with torch.set_grad_enabled(False):
            embeddings, logits = model(imgs)
            outputs["embeddigs"].append(embeddings)
            outputs["logits"].append(logits)
    for k,v in outputs:
        outputs[k] = torch.cat(v, 0)
    return outputs


def evaluate_logits(labels, logits, class_names):
    evaluations = {}
    evaluations["logits_top_1_acc"] = top_k_accuracy_score(labels, logits, k=1, labels=class_names)
    evaluations["logits_top_5_acc"] = top_k_accuracy_score(labels, logits, k=5, labels=class_names)
    evaluations["logits_macro_avg_prec"] = average_precision_score(labels, logits, average='macro')
    evaluations["logits_micro_avg_prec"] = average_precision_score(labels, logits, average='micro')
    return evaluations

def evaluate_dist(labels, embeddings, ref_labels=None, ref_embedding=None, distance_type="euclidean"):
    if ref_labels is not None:
        assert ref_embedding is not None
        similar_label_matrix = (labels.reshape(-1,1) == ref_labels)
        distance_matrix = pairwise_distances(embeddings, ref_embeddings, distance_type=distance_type)
        n = len(ref_labels)
    else:
        n = len(labels)
        similar_label_matrix = (labels.reshape(-1,1) == labels)
        shifted_similar_labels = np.roll(similar_label_matrix, -1, 1)
        similar_label_matrix = np.where(np.tri(n).T > 0, shifted_similar_labels, shifted_similar_labels)
        distance_matrix = pairwise_distances(embeddings, distance_type=distance_type)
        shifted_distances = np.roll(distance_matrix, -1, 1)
        distance_matrix = np.where(np.tri(n).T > 0, shifted_distances, distance_matrix)
    evaluations = {}
    evaluations["dist_lraps"] = label_ranking_average_precision_score(similar_label_matrix, -distance_matrix)
    sorted_distances_inds = distance_matrix.argsort(axis=1)
    correct_labels_ranked = np.take_along_axis(similar_label_matrix, sorted_distances_inds, axis=1)
    evaluations["dist_avg_top_1_acc"] = correct_labels_ranked[:,:1].sum(axis=1).mean().item() # sample avg
    evaluations["dist_avg_top_1_acc"] = (correct_labels_ranked[:,:5].sum(axis=1)/5).mean().item() # sample avg
    sample_precisions = (correct_labels_ranked.cumsum(axis=1) * correct_labels_ranked)/np.arange(1, n+1)
    N = correct_labels_ranked.sum(axis=1)
    evaluations["dist_avg_map_1"] = (sample_precisions[:,:1].sum(axis=1)/N).mean().item() # sample avg
    evaluations["dist_avg_map_5"] = (sample_precisions[:,:5].sum(axis=1)/N).mean().item() # sample avg
    return evaluations

def evaluate_query_ref(model, query_outputs, ref_outputs, class_names, use_side_labels=False):
    logits_eval = evaluate_logits(query_outputs["label"], query_outputs["logits"], class_names)
    dist_eval = evaluate_dist(query_outputs["label"], query_outputs["embeddings"], ref_outputs["label"], ref_outputs["embeddings"])
    return {**logits_eval, **dist_eval}