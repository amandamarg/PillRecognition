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
        val_counts = self.df[self.labelcol].value_counts()
        val_counts = val_counts[val_counts >= self.min_per_class]
        valid_classes = self.df[self.df[self.labelcol].isin(val_counts.index.tolist())][self.labelcol].unique() #dropping any labels with less than self.min_per_class instances
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        self.refs_per_class = refs_per_class
        if self.refs_per_class > 0:
            assert refs_per_class < self.min_per_class
            valid_classes = np.intersect1d(valid_classes, df[df.is_ref][self.labelcol].unique())
            self.refs = {l: self.df[self.df.is_ref & (self.df[self.labelcol] == l)].index.tolist() for l in valid_classes}
        else:
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
        
            # add new classes until min_classes in batch
            class_choices = np.setdiff1d(list(unseen.keys()), curr_batch_labels)
            if len(class_choices) < self.min_classes:
                class_choices = class_choices + np.rng.choice(np.setdiff1d(list(seen.keys()), curr_batch_labels), self.min_classes - len(class_choices), replace=False)
            
            for curr_label in class_choices:
                indicies = []
                if self.refs is not None:
                    indicies.extend(self.refs[curr_label])
                if curr_label in unseen.keys():
                    indicies.extend(self.rng.choice(unseen[curr_label], min(min_per_class, len(unseen[curr_label])), replace=False))
                    self.update_seen_unseen(seen, unseen, curr_label, indicies)
                if len(indicies) < self.min_per_class:
                    indicies.extend(self.rng(seen[curr_label], self.min_per_class-len(indicies), replace=False))
                curr_batch.extend(curr_label)
                curr_batch_labels.append(curr_label)

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
        return len(self.df)//self.batch_size
    
class CustomBatchSamplerPillID2(BatchSampler):
    def __init__(self, df, batch_size, labelcol="label", generator=None, min_per_class=2, min_classes=2, batch_size_mode=None, debug=False):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        self.min_per_class = min_per_class # exclude any classes that don't have at least min_per_class instances
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        if shuffle:
            self.valid_inds = {k: self.rng.choice(v.values, len(v), replace=False) for k,v in self.df.groupby(self.labelcol).groups.items() if len(v) >= self.min_per_class}
        else:
            self.valid_inds = {k: v.values for k,v in self.df.groupby(self.labelcol).groups.items() if len(v) >= self.min_per_class}
        # self.labels = []
        # self.inds = []
        # for group_name, group_inds in self.df.groupby(self.labelcol).groups.items():
        #     self.labels.extend([group_name] * len(group_inds))
        #     self.inds.extend(group_inds.values)
        self.valid_classes = self.df.value_counts(self.labelcol)[self.df.value_counts(self.labelcol) >= self.min_per_class].index.values
        self.inds = self.df[self.df[self.labelcol].isin(self.valid_classes)].index.values
        self.labels = self.df[self.labelcol].iloc[self.inds]
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
    
    def add_new_class(self, curr_batch, label_map):
        batch_labels = self.labels[curr_batch].unique()
        label_mask = np.isin(self.labels, batch_labels)
        

    def __iter__(self):
        labeled_inds = np.equal.outer(self.labels, self.valid_classes)
        used = np.zeros((len(self.inds),1))
        numbered_inds = np.arange(len(self.inds))
        while (used == 0).any():
            curr_batch = []
            unused_masked = labeled_inds & (used == 0)
            class_options = np.argwhere(np.sum(unused_masked, axis=0) >= self.min_per_class)
            classes = self.rng.choice(class_options, min(self.min_classes, len(class_options)), replace=False)
            if len(classes) < self.min_classes:
                class_options = np.setdiff1d(np.arange(len(self.valid_classes)), classes.flatten())
                classes = classes = self.rng.choice(class_options, self.min_classes - len(classes), replace=False)
            for curr_label in classes:
                unused_inds = np.argwhere(unused_masked)
                selected_inds = self.rng.choice(unused_inds, min(len(unused_inds), self.min_per_class), replace=False)
                if len(selected_inds) < self.min_per_class:
                    used_inds = np.argwhere(labeled_inds[:, curr_label] & (used > 0))
                    selected_inds = selected_inds + self.rng.choice(used_inds, self.min_per_class - len(selected_inds), replace=False)
                used[selected_inds] += 1
                curr_batch.extend(selected_inds)
            
            
            while len(curr_batch) < self.batch_size:
                if (self.batch_size - len(curr_batch)) < self.min_per_class:
                    x = np.where(self.labels.isin(batch_labels), x, 0)
                else:
                    x = np.where(self.labels.isin(batch_labels) | (x >= self.min_per_class), x, 0)
                if len(np.nonzero(x)) == 0:
                    break
                curr_label = self.rng.choice(np.nonzero(x))
                selected_inds = np.argwhere(unused_masked[:,curr_label], min(x[curr_label.item()], self.batch_size - len(curr_batch)), replace=False)
                used[selected_inds] += 1
                curr_batch.extend(selected_inds)
            
            unused_masked = labeled_inds & (used == 0)
            batch_labels = self.labels[curr_batch].unique()
            x = np.sum(unused_masked, axis=0)
            label_mask = np.argwhere(x < self.min_per_class)
            assert self.valid_classes[label_mask].isin(batch_labels).all()
            leftovers = np.argwhere(np.isin(self.labels, self.valid_classes[np.argwhere(x < self.min_per_class)]))
            selected_inds = self.rng.choice(leftovers, min(len(leftovers), self.batch_size - len(curr_batch)),replace=False)
            used[selected_inds] += 1
            curr_batch.exted(selected_inds)
            size_diff = self.batch_size - len(curr_batch)

            
                
            if size_diff > 0 and self.batch_size_mode in ['min', 'strict']:
                curr_batch_mask = np.isin(np.arange(len(self.inds)), curr_batch)
                batch_labels = self.labels[curr_batch_mask].unique()
                curr_batch_label_mask = np.isin(self.labels,batch_labels)
                unused_classes = self.valid_classes[~np.isin(self.valid_classes, batch_labels)]


                # np.argwhere(curr_batch_label_mask & ~curr_batch_mask)
                # diff_classes = np.argwhere(~curr_batch_label_mask & ~curr_batch_mask)
                


            if size_diff < 0 and self.batch_size_mode in ['max', 'strict']:
                curr_batch = curr_batch[:size_diff]
            # class_options = np.argwhere(x >= np.where(np.isin(self.labels, batch_labels), 1, self.min_per_class))
            curr_batch = self.inds[curr_batch]
            yield curr_batch
            
    def __len__(self):
        return len(self.df)//self.batch_size
    

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
        val_counts = all_imgs_df['label'].value_counts()
        val_counts = val_counts[val_counts >= min_per_class]
        max_classes = len(val_counts)
        max_batch_size = all_imgs_df['label'].isin(val_counts.index.tolist()).sum()
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
    