import os
from glob import glob
import torch
import pandas as pd
import tqdm
import numpy as np
from dataset import PillImages, TwoSidedPillImages, CustomBatchSamplerPillID
from torch.utils.data import DataLoader


def load_data(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', img_dir='classification_data'):
    csv_files = glob(data_root_dir + '/folds/**/*.csv', recursive=True)
    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    folds = sorted([x for x in csv_files if not x.endswith("all.csv")])
    all_imgs_df = pd.read_csv(all_imgs_csv)
    fold_indicies = [np.where(all_imgs_df.image_path.isin(pd.read_csv(fold).image_path))[0] for fold in folds]
    all_imgs_df['image_path'] = all_imgs_df['image_path'].apply(lambda x: os.path.join(data_root_dir, 'classification_data', x))
    return all_imgs_df, fold_indicies

def split_data(all_imgs_df, fold_indicies, val_fold=3, test_fold=4):
    val_df = all_imgs_df.iloc[fold_indicies[val_fold]].reset_index(drop=True)
    test_df = all_imgs_df.iloc[fold_indicies[test_fold]].reset_index(drop=True)
    train_df = all_imgs_df.iloc[np.concatenate([f for i,f in enumerate(fold_indicies) if i != 3 and i != 4])].reset_index(drop=True)
    return {'train': train_df,'val': val_df, 'test': test_df}

def front_back_pairs(df, labelcol):
    pairs = []
    for group_label,group in df.groupby(labelcol):
        front = group[group.is_front]
        back = group[~group.is_front]
        k = min(len(front), len(back))
        if k > 0:
            pairs.extend(list(zip(front.iloc[:k].index.to_numpy(), back.iloc[:k].index.to_numpy())))
    return np.array(pairs)

def get_datasets(partitions, ref_df, labelcol, two_sided, **kwargs):
    datasets = {}
    if two_sided:
        ref_pairs = front_back_pairs(ref_df, labelcol)
        front_ref = ref_df.iloc[ref_pairs[:,0]]
        back_ref = ref_df.iloc[ref_pairs[:,1]]
        for k,v in partitions.items():
            cons_pairs = front_back_pairs(v, labelcol)
            front = pd.concat([v.iloc[cons_pairs[:,0]], front_ref])
            back = pd.concat([v.iloc[cons_pairs[:,1]], back_ref])
            datasets[k] = TwoSidedPillImages(front_df=front, back_df=back, phase=k, labelcol=labelcol, **kwargs)
    else:
        for k,v in partitions.items():
            datasets[k] = PillImages(df=pd.concat([v, ref_df]), phase=k, labelcol=labelcol, **kwargs)
    return datasets


def save_model(models, model_name, curr_epoch, save_dir = '/Users/Amanda/Desktop/PillRecognition/model'):
    os.makedirs(os.path.join(save_dir, model_name, 'embedding'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, model_name, 'classifier'), exist_ok=True)

    torch.save(models['embedding'], os.path.join(save_dir, model_name, 'embedding', 'em_epoch_{:d}.pt'.format(curr_epoch)))
    torch.save(models['classifier'], os.path.join(save_dir, model_name, 'classifier', 'cl_epoch_{:d}.pt'.format(curr_epoch)))

def embed_all(models_dict, dataloader, embedding_size, n_classes, device, include_logits=True, sort=False):
    num_imgs = len(dataloader.dataset)
    start_idx = 0

    all_labels = torch.zeros(num_imgs).to(device)
    all_embeddings = torch.zeros(num_imgs, embedding_size).to(device)

    models_dict['embedding'].eval()

    if include_logits:
        models_dict['classifier'].eval()
        all_logits = torch.zeros(num_imgs, n_classes).to(device)

    with torch.set_grad_enabled(False):
        for data in tqdm(dataloader, total=len(dataloader)):
            imgs = data[0].to(device)
            labels = data[1].to(device)
            end_idx = start_idx + len(labels)
            all_labels[start_idx:end_idx] = labels


            embeddings =  models_dict['embedding'](imgs)
            all_embeddings[start_idx:end_idx, :] = embeddings

            if include_logits:
                logits = models_dict['classifier'](embeddings)
                all_logits[start_idx:end_idx, :] = logits

            start_idx = end_idx
    if sort:
        all_labels, sorted_ind = all_labels.sort()
        all_embeddings = all_embeddings[sorted_ind]
        if include_logits:
            all_logits = all_logits[sorted_ind]

    if include_logits:
        return (all_labels.type(torch.int32), all_embeddings, all_logits)
    return (all_labels.type(torch.int32), all_embeddings)

def get_triplets(labels, ref_labels=None):
    if ref_labels is None:
        one_hot_labels = np.equal.outer(labels, labels)
        pos_pairs = np.argwhere(one_hot_labels)
        pos_pairs = pos_pairs[pos_pairs[:,0] != pos_pairs[:,1]]
    else:
        one_hot_labels = np.equal.outer(labels, ref_labels)
        pos_pairs = np.argwhere(one_hot_labels)
    neg_pairs = np.argwhere(~one_hot_labels)
    same_anchors = np.argwhere(np.equal.outer(pos_pairs[:,0], neg_pairs[:,0]))
    triplets = np.column_stack((pos_pairs[same_anchors[:,0]], neg_pairs[same_anchors[:,1]][:,1]))
    return triplets
    