from dataset import TwoSidedPillImages, PillImages
from glob import glob
import pandas as pd
import os
import numpy as np

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
    train_df = all_imgs_df.iloc[np.concatenate([f for i,f in enumerate(fold_indicies) if i != val_fold and i != test_fold])].reset_index(drop=True)
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
