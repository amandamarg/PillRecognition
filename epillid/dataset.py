from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import utils
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter

class PillImages(Dataset):
    def __init__(self, df, phase, transform=None, augment=None, labelcol="pilltype_id", label_encoder=None):
        self.df = df
        self.phase = phase
        self.transform = transform
        self.augment = augment
        self.labelcol = labelcol
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.load_img(row.image_path)

        if self.transform is not None:
            img = self.transform(img)

        label = row[self.labelcol]
        if self.label_encoder is not None:
            label = self.label_encoder.transform([label])[0]
        return label, img, row.is_front, row.is_ref
    
    def load_img(self, img_path):
        if not os.path.exists(img_path):
            print("img not found", img_path)
            return
        to_tensor = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.float32, scale=True)])
        return to_tensor(Image.open(img_path))

class TwoSidedPillImages(Dataset):
    def __init__(self, front_df, back_df, phase, transform=None, augment=None, labelcol="pilltype_id", label_encoder=None):
        assert len(front_df) == len(back_df)
        self.front_df = front_df
        self.back_df = back_df
        self.phase = phase
        self.transform = transform
        self.augment = augment
        assert (self.front_df[labelcol].values == self.back_df[labelcol].values).all()
        self.labelcol = labelcol
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.front_df)
    
    def __getitem__(self, index):
        front_row = self.front_df.iloc[index]
        back_row = self.back_df.iloc[index]

        front_img = self.load_img(front_row['image_path'])
        back_img = self.load_img(back_row['image_path'])

        if self.transform is not None:
            front_img = self.transform(front_img)
            back_img = self.transform(back_img)

        label = front_row[self.labelcol]
        if self.label_encoder is not None:
            label = self.label_encoder.transform([label])[0]
        return label, (front_img, back_img), (front_row.is_ref, back_row.is_ref)
    
    def load_img(self, img_path):
        if not os.path.exists(img_path):
            print("img not found", img_path)
            return
        to_tensor = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.float32, scale=True)])
        return to_tensor(Image.open(img_path))
    
class CustomBatchSamplerPillID(BatchSampler):
    def __init__(self, df, batch_size, labelcol="pilltype_id", generator=None, min_per_class=2, min_classes=2, batch_size_mode=None, refs_per_class=0, debug=False):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        self.min_per_class = min_per_class # drops any classes that don't have at least min_per_class instances
        valid_classes = self.df[self.labelcol].value_counts().where(lambda x: x >= self.min_per_class).index.tolist() #dropping any labels with less than self.min_per_class instances
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        self.refs_per_class = refs_per_class
        if self.refs_per_class > 0:
            assert refs_per_class < self.min_per_class
            valid_classes = np.intersect1d(valid_classes, df[df.is_ref][self.labelcol].unique())
            self.df = self.df[self.df[self.labelcol].isin(valid_classes)]
            self.refs = {k: self.rng.choice(v.values, self.refs_per_class, replace=False) for k,v in self.df[self.df.is_ref].groupby(self.labelcol).groups.items() if len(v) >= self.refs_per_class}
        else:
            self.df = self.df[self.df[self.labelcol].isin(valid_classes)]
            self.refs_per_class = 0
            self.refs = None
        assert min_classes <= len(valid_classes)
        assert (min_classes * min_per_class) <= batch_size
        assert len(self.df) >= batch_size
        self.min_classes = min_classes # requires at least min_classes classes in each batch (may repeat towards end if not enough classes remaining)
        self.batch_size_mode = batch_size_mode
        '''
        max: batches will be at most self.batch_size, but may be smaller
        min: batches will be a minimum of self.batch_size, but may be larger
        strict: batches will be exactly self.batch_size, but may break min_per_class condition
        None: will try to make batches of size self.batch_size but may be more or less as needed
        '''
        assert batch_size_mode in ["max", "min", "strict", None]
        self.debug = debug

    def verify_batchsize(self, curr_batch):
        if self.batch_size_mode == "max":
            return len(curr_batch) <= self.batch_size
        elif self.batch_size_mode == "min":
            return len(curr_batch) >= self.batch_size
        elif self.batch_size_mode == "strict":
            return len(curr_batch) == self.batch_size
        else:
            return True

        
    def update_seen_unseen(self, seen, unseen, label, inds):
        unseen[label] = np.setdiff1d(unseen[label], inds)
        if len(unseen[label]) == 0:
            unseen.pop(label)
        if label not in seen.keys():
            seen[label] = []
        seen[label].extend(inds)
        

    # def get_min_size(self, label):
    #     return self.min_per_class if self.refs is None else (self.min_per_class - len(self.refs[label]))

    # def add_to_curr_batch(self, curr_batch, curr_batch_labels, label, ind_dict):
    #     if label not in curr_batch_labels:
    #         if self.refs is not None:
    #             curr_batch.extend(self.refs[label])
    #         inds_to_add = self.rng.choice(ind_dict[label], self.min_per_class - self.refs_per_class, replace=False)
    #         curr_batch_labels.append(label)
    #     else:
    #         inds_to_add = self.rng.choice(ind_dict[label], 1, replace=False)
    #     curr_batch.extend(inds_to_add)
    #     return inds_to_add

    
        
    def __iter__(self):
        #maybe just shuffle once at beginning instead of always using rng.choice
        min_per_class = self.min_per_class - self.refs_per_class
        if self.refs is None:
            unseen = {k:v.values for k,v in self.df.groupby(self.labelcol).groups.items()}
        else:
            unseen = {k:v.values for k,v in self.df[~self.df.is_ref].groupby(self.labelcol).groups.items()}
        seen = {}

        while len(unseen) > 0:
            curr_batch = []
            curr_batch_labels = []
            while len(curr_batch_labels) < self.min_classes:
                # add new classes until min_classes in batch
                class_choices = np.setdiff1d(list(unseen.keys()), curr_batch_labels)
                add_seen = (len(class_choices) == 0)
                if add_seen:
                    #if no classes left with unseen images, add classes that were already seen and not in batch
                    curr_label = self.rng.choice(np.setdiff1d(list(seen.keys()), curr_batch_labels))
                    indicies = self.rng.choice(seen[curr_label], min_per_class, replace=False)
                else:
                    curr_label = self.rng.choice(class_choices)
                    indicies = unseen[curr_label]
                    indicies = self.rng.choice(indicies, min_per_class, replace=False)
                    self.update_seen_unseen(seen, unseen, curr_label, indicies)
                if self.refs is not None:
                    curr_batch.extend(self.refs[curr_label])
                curr_batch.extend(indicies)
                curr_batch_labels.append(curr_label)
            assert len(curr_batch_labels) >= self.min_classes    
            size_diff = self.batch_size - len(curr_batch)
            while size_diff > 0:
                class_choices = list(unseen.keys()) if size_diff >= self.min_per_class else np.intersect1d(list(unseen.keys()), curr_batch_labels)
                if len(class_choices) == 0:
                    break
                
                curr_label = self.rng.choice(class_choices)
                indicies = unseen[curr_label]
                if curr_label in curr_batch_labels:
                    indicies = self.rng.choice(indicies, min(min_per_class, len(indicies)), replace=False)
                else:                    
                    indicies = self.rng.choice(indicies, min_per_class, replace=False)
                self.update_seen_unseen(seen, unseen, curr_label, indicies)
                
                
                if curr_label not in curr_batch_labels:
                    if self.refs is not None:
                        curr_batch.extend(self.refs[curr_label])
                    curr_batch_labels.append(curr_label)
                    
                curr_batch.extend(indicies)
                size_diff = self.batch_size - len(curr_batch)
                
            # leftovers = [list(unseen.pop(l)) for l in np.intersect1d(list(unseen.keys()), curr_batch_labels) if len(unseen[l]) < min_per_class]
            # leftovers = sum(leftovers, [])
            for l in np.intersect1d(list(unseen.keys()), curr_batch_labels):
                if len(unseen[l]) < min_per_class:
                    indicies = unseen.pop(l)
                    seen[l].extend(indicies)
                    curr_batch.extend(indicies)
            size_diff = self.batch_size - len(curr_batch)

            if size_diff > 0 and self.batch_size_mode in ['min', 'strict']:
                available = [list(np.setdiff1d(seen[l], curr_batch)) for l in curr_batch_labels]
                available = sum(available, [])
                curr_batch.extend(self.rng.choice(available, min(size_diff, len(available)), replace=False))
                size_diff = self.batch_size - len(curr_batch)
                while size_diff > 0:
                    class_choices = np.setdiff1d(list(seen.keys()), curr_batch_labels)
                    curr_label = self.rng.choice(class_choices)
                    if self.refs is not None:
                        curr_batch.extend(self.refs[curr_label])
                    curr_batch_labels.append(curr_label)
                    indicies = seen[curr_label]
                    indicies = self.rng.choice(indicies, min_per_class, replace=False)
                    curr_batch.extend(indicies)
                    size_diff = self.batch_size - len(curr_batch)
            if size_diff < 0 and self.batch_size_mode in ['max', 'strict']:
                curr_batch = curr_batch[:size_diff]

            if self.debug:
                assert self.verify_batchsize(curr_batch)
                assert len(set(curr_batch_labels)) >= self.min_classes
                assert len(list(set(curr_batch))) == len(curr_batch)
                for l in curr_batch_labels:
                    inds = seen[l]
                    if self.refs is not None:
                        for i in self.refs[l]:
                            assert i in curr_batch
                        inds = inds + self.refs[l]  
                    if self.batch_size_mode != 'strict':
                        assert len(np.intersect1d(inds, curr_batch)) >= self.min_per_class
            yield curr_batch

    def __len__(self):
        return len(self.df)

if __name__ == "__main__":
    all_imgs_df, fold_indicies = utils.load_data()
    unique_classes = all_imgs_df['label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)
    ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    # partitions = utils.split_data(all_imgs_df, fold_indicies)
    # datasets = utils.get_datasets(partitions, ref_df, labelcol='label', two_sided=False, label_encoder = label_encoder)
    # train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=32, labelcol='label', refs_per_class=2, min_per_class=3)
    # val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=32, labelcol='label', min_per_class=2, batch_size_mode='min')
    # dataloaders = {'train': DataLoader(datasets['train'], batch_sampler=train_batch_sampler), 'val': DataLoader(datasets['val'], batch_sampler=val_batch_sampler)}
    # min_per_class=2, min_classes=2, batch_size_mode=None, refs_per_class=0
    sampler_args = []

    all_img_dataset = PillImages(all_imgs_df, phase='all', labelcol='label', label_encoder=label_encoder)
    min_per_class_range = np.linspace(2, all_imgs_df['label'].value_counts().unique().max(), 2, dtype=int)
    for min_per_class in min_per_class_range:
        max_classes = (all_imgs_df['label'].value_counts() >= min_per_class).sum()
        max_batch_size = all_imgs_df['label'].isin((all_imgs_df['label'].value_counts() >= min_per_class).index.tolist()).sum()
        for min_classes in np.linspace(2, max_classes, 2, dtype=int):
            min_batch_size = min_classes*min_per_class                
            for batch_size in np.linspace(min_batch_size, max_batch_size, 3, dtype=int):
                sampler_args.append((min_per_class, min_classes, batch_size))
    print(len(sampler_args))
    for min_per_class, min_classes, batch_size in sampler_args:
        print("min_per_class={}, min_classes={}, batch_size:{}".format(min_per_class, min_classes, batch_size))
        sampler = CustomBatchSamplerPillID(all_imgs_df, batch_size=batch_size, labelcol='label', min_classes=min_classes, min_per_class=min_per_class, batch_size_mode=None, debug=True)
        # dataloader = DataLoader(all_img_dataset, batch_sampler=sampler)
        # for _ in tqdm(dataloader, total=len(dataloader)):
        #     continue
        for _ in tqdm(sampler, total=len(sampler)):
            continue
    # min_per_class_range = range(2, all_imgs_df['label'].value_counts().unique().max() + 1)
    # cons_with_refs = all_imgs_df[all_imgs_df['label'].isin(all_imgs_df[all_imgs_df.is_ref]['label'].unique()) & ~all_imgs_df.is_ref]
    # min_per_class_range = range(2, cons_with_refs['label'].value_counts().unique().max()+1)
    # for min_per_class in min_per_class_range:
    #     max_classes = (cons_with_refs['label'].value_counts() >= min_per_class).sum()
    #     for min_classes in range(2, max_classes+1):
    #         for mode in ['min', 'max', None, 'strict']:
    #             samplers.append(CustomBatchSamplerPillID(all_imgs_df, batch_size=32, labelcol='label', min_classes=min_classes, min_per_class=min_per_class + 2, refs_per_class=2, batch_size_mode=mode))

    # for m in modes:
    #     datasets['train'].df['label'].value_counts().unique()
        
        # CustomBatchSamplerPillID(datasets['train'].df, batch_size=32, labelcol='label', refs_per_class=2, min_per_class=3)
    