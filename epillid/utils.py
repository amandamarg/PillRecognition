import os
from glob import glob
import torch
import pandas as pd
import tqdm
import numpy as np
from dataset import PillImages

def load_data(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', img_dir='classification_data'):
    csv_files = glob(data_root_dir + '/folds/**/*.csv', recursive=True)
    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    folds = sorted([x for x in csv_files if not x.endswith("all.csv")])
    all_imgs_df = pd.read_csv(all_imgs_csv)
    fold_indicies = [np.where(all_imgs_df.image_path.isin(pd.read_csv(fold).image_path))[0] for fold in folds]
    return all_imgs_df, fold_indicies

def split_data(all_imgs_df, fold_indicies, val_fold=3, test_fold=4):
    val_df = all_imgs_df.iloc[fold_indicies[val_fold]].reset_index(drop=True)
    test_df = all_imgs_df.iloc[fold_indicies[test_fold]].reset_index(drop=True)
    train_df = all_imgs_df.iloc[np.concatenate([f for i,f in enumerate(fold_indicies) if i != 3 and i != 4])].reset_index(drop=True)
    return {'train': train_df,'vale': val_df, 'test': test_df}

def get_dataset(cons_df, ref_df, phase, labelcol):
    return PillImages(pd.concat([cons_df, ref_df]), phase, labelcol=labelcol)

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

