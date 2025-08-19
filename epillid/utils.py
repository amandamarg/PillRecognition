import os
from glob import glob
import torch
import pandas as pd
import tqdm

def get_csv(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', val_fold=3, test_fold=4):
    csv_files =  glob(data_root_dir + '/folds/**/*.csv', recursive=True)
    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    folds = sorted([x for x in csv_files if not x.endswith("all.csv")])
    return all_imgs_csv, folds

# def load_data(all_imgs_csv, folds, folds_sorted=True)
#     if not folds_sortedÑ
#         folds=sorted(folds)
#     all_imgs_df = pd.read_csv(all_imgs_csv)
#     test_df = pd.read_csv(folds[test_fold])
#     val_df = pd.read_csv(folds[val_fold])

#     img_dir = 'classification_data'
#     for df in [all_images_df, val_df, test_df]:
#         df['image_path'] = df['image_path'].apply(lambda x: os.path.join(data_root_dir, img_dir, x))

#     val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
#     train_df = all_imgs_df[~all_imgs_df.isin(val_test_image_paths)].reset_index()
    
#     return {'train': train_df, 'val': val_df, 'test': test_df, 'all': all_imgs_df}


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

def save_best_epoch()