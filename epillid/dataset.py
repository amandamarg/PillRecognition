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
    def __init__(self, df, batch_size, labelcol="pilltype_id", generator=None, min_per_class=2, min_classes=2, batch_size_mode=None, refs_per_class=0, keep_remainders=False, debug=False):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        self.min_per_class = min_per_class # drops any classes that don't have at least min_per_class instances
        val_counts = self.df.value_counts(self.labelcol)
        valid_classes = val_counts[val_counts >= self.min_per_class].index.tolist()
        
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        self.refs_per_class = refs_per_class
        if self.refs_per_class > 0:
            assert refs_per_class < self.min_per_class
            val_counts = self.df[self.df.is_ref].value_counts(self.labelcol)
            valid_ref_classes = val_counts[val_counts >= self.refs_per_class].index.tolist()
            valid_classes = np.intersect1d(valid_classes, valid_ref_classes)
            self.refs = {k: v.values[:self.refs_per_class] for k,v in self.df[self.df.is_ref].groupby(self.labelcol).groups.items() if k in valid_classes}
        else:
            self.refs_per_class = 0
            self.refs = None
        self.valid_classes = valid_classes
        assert min_classes <= len(valid_classes)
        assert (min_classes * min_per_class) <= batch_size
        assert len(self.df) >= batch_size
        self.min_classes = min_classes # requires at least min_classes classes in each batch (may repeat towards end if not enough classes remaining)
        self.batch_size_mode = batch_size_mode
        '''
        max: batches will be at most self.batch_size, but may be smaller
        min: batches will be a minimum of self.batch_size, but may be larger
        strict: batches will be exactly self.batch_size, but may break min_per_class condition*
        None: will try to make batches of size self.batch_size but may be more or less as needed

        *in the case of 'strict', min_per_class will be broken for at most one class and any inds for that class where min_per_class are broken will either have been already seen or will be reused in a future batch
        '''
        assert batch_size_mode in ["max", "min", "strict", None]
        self.keep_remainders = keep_remainders
        ''' 
        if, after adding indicies to the curr_batch, the remaining indicies are less than self.min_per_batch, they will be handled according to this argument
        if self.keep_remainders is True, the remainders will be kept as unseen until they can be added to a class, in which case the class will use seen indicies to fufill min_per_class
        if self.keep_remainders is False, the remainders will be marked as seen and ignored
        if batch_size_mode is min or None, keep_remainders doesn't matter because all remainders will be added to the batch regardless
        '''
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
        if label not in seen.keys():
            seen[label] = []
        seen[label].extend(inds)
        if len(unseen[label]) == 0:
            unseen.pop(label)
            return []
        return unseen[label]
    
    def cleanup_leftovers(self, leftovers, seen, unseen, curr_batch):
        for k,v in leftovers.items():
            size_diff = self.batch_size - len(curr_batch)
            if len(v) <= size_diff or self.batch_size_mode in ['min', None]:
                curr_batch.extend(v)
                self.update_seen_unseen(seen, unseen, k, v)
            elif size_diff > 0:
                inds = self.rng.choice(v, size_diff, replace=False)
                curr_batch.extend(inds)
                remaining = self.update_seen_unseen(seen, unseen, k, inds)
                if len(remaining) > 0 and not self.keep_remainders:
                    self.update_seen_unseen(seen, unseen, k, remaining)
            elif not self.keep_remainders:
                self.update_seen_unseen(seen, unseen, k, v)
            else:
                return
            
    
    def grow_existing_classes(self, curr_batch, batch_labels, unseen, seen, add_from_seen=False):
        size_diff = self.batch_size - len(curr_batch)
        if size_diff <= 0:
            return
        
        if add_from_seen:
            present_labels = np.intersect1d(list(seen.keys()), batch_labels)
        else:
            present_labels = np.intersect1d(list(unseen.keys()), batch_labels)

        if len(present_labels) == 0:
            return
            
        leftovers = {}
        min_per_class = (self.min_per_class - self.refs_per_class)
        for label in present_labels:
            if add_from_seen:
                inds = np.setdiff1d(seen[label], curr_batch)
            else:
                inds = unseen[label]
            selected_inds = self.rng.choice(inds, min(len(inds), size_diff), replace=False)
            if not add_from_seen:
                remaining = self.update_seen_unseen(seen, unseen, label, selected_inds)
                if len(remaining) < min_per_class and len(remaining) > 0:
                    leftovers[label] = remaining
            curr_batch.extend(selected_inds)
            size_diff = self.batch_size - len(curr_batch)
            if size_diff <= 0:
                break
        if len(leftovers) > 0:
            self.cleanup_leftovers(leftovers, seen, unseen, curr_batch)
            

            
    def grow_new_classes(self, curr_batch, batch_labels, unseen, seen, seen_is_default=False, default_only=True, num_classes=None, update_seen_unseen=True):
        if num_classes is None:
            num_classes = (self.batch_size - len(curr_batch)) // self.min_per_class

        if num_classes <= 0:
            return 0
        min_per_class = (self.min_per_class - self.refs_per_class)
            
        default = list(np.setdiff1d(list(seen.keys()), batch_labels)) if seen_is_default else list(np.setdiff1d(list(unseen.keys()), batch_labels)) 
        if len(default) < num_classes and not default_only:
            add_classes = default
            backup = list(unseen.keys()) if seen_is_default else list(seen.keys())
            backup = list(np.setdiff1d(backup, default + batch_labels))
            add_backup = self.rng.choice(backup, min(num_classes - len(default), len(backup)), replace=False)
            add_classes.extend(add_backup)
        else:
            add_classes = list(self.rng.choice(default, min(len(default), num_classes), replace=False))

        if len(add_classes) == 0:
            return 0
        
        leftovers = {}        
        for label in add_classes:
            assert label not in batch_labels
            inds = []
            if self.refs is not None:
                curr_batch.extend(self.refs[label])
            if label in unseen.keys():
                inds.extend(self.rng.choice(unseen[label], min(len(unseen[label]), min_per_class), replace=False))
                if update_seen_unseen:
                    remaining = self.update_seen_unseen(seen, unseen, label, inds)
                    if len(remaining) < min_per_class and len(remaining) > 0:
                        leftovers[label] = remaining
            
            if len(inds) < min_per_class:
                ind_choices = seen[label]
                inds.extend(self.rng.choice(ind_choices, min_per_class-len(inds), replace=False))
            assert len(inds) == min_per_class
            curr_batch.extend(inds)
            batch_labels.append(label)

        if len(leftovers) > 0:
            self.cleanup_leftovers(leftovers, seen, unseen, curr_batch)
        
        return len(add_classes)
    
            
    def __iter__(self):
        #maybe just shuffle once at beginning instead of always using rng.choice
        if self.refs is None:
            unseen = {k:v.values for k,v in self.df.groupby(self.labelcol).groups.items() if k in self.valid_classes}
        else:
            unseen = {k:v.values for k,v in self.df[~self.df.is_ref].groupby(self.labelcol).groups.items() if k in self.valid_classes}
        seen = {}
        

        while len(unseen) > 0:
            curr_batch = []
            curr_batch_labels = []
            self.grow_new_classes(curr_batch, curr_batch_labels, unseen, seen, seen_is_default=False, default_only=False, num_classes=self.min_classes) # make sure we reach minimum number of classes by first adding from unseen, and then from seen only if needed
            self.grow_new_classes(curr_batch, curr_batch_labels, unseen, seen, seen_is_default=False, default_only=True) # add as many unseen classes as we can without going over batch size
            self.grow_existing_classes(curr_batch, curr_batch_labels, unseen, seen, add_from_seen=False)


            if len(curr_batch) < self.batch_size and self.batch_size_mode in ['min', 'strict']:
                if (self.batch_size - len(curr_batch)) > self.min_per_class:
                    self.grow_new_classes(curr_batch, curr_batch_labels, unseen, seen, seen_is_default=True, default_only=True)
                self.grow_existing_classes(curr_batch, curr_batch_labels, unseen, seen, add_from_seen=True)
                is_strict = (self.batch_size_mode == 'strict')
                if len(curr_batch) < self.batch_size:
                    # at this point, it should be guaranteed that (self.batch_size - len(curr_batch)) < self.min_per_class so adding 1 more class will put us over the self.batch_size
                    self.grow_new_classes(curr_batch, curr_batch_labels, unseen, seen, seen_is_default=is_strict, default_only=False, num_classes=1, update_seen_unseen=(not is_strict))
                
            if len(curr_batch) > self.batch_size and self.batch_size_mode in ['max', 'strict']:
                curr_batch = curr_batch[:self.batch_size]

            if self.debug:
                assert self.verify_batchsize(curr_batch)
                assert len(set(curr_batch_labels)) == len(curr_batch_labels)
                assert len(set(curr_batch_labels)) >= self.min_classes
                val_counts = self.df.iloc[curr_batch][self.labelcol].value_counts()
                if self.batch_size_mode == 'strict':
                    assert len(val_counts[val_counts < self.min_per_class]) <= 1
                else:
                    assert (val_counts >= self.min_per_class).all()
            yield curr_batch

    def __len__(self):
        return len(self.df[self.df[self.labelcol].isin(self.valid_classes)])//self.batch_size



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

    for mode in ['min', 'max', 'strict', None]:
        print("{}:".format(mode if mode is not None else 'None'))
        for min_per_class, min_classes, batch_size in sampler_args:
            print("min_per_class={}, min_classes={}, batch_size:{}".format(min_per_class, min_classes, batch_size))
            sampler = CustomBatchSamplerPillID(all_imgs_df, batch_size=batch_size, labelcol='label', min_classes=min_classes, min_per_class=min_per_class, keep_remainders=True, batch_size_mode=mode, debug=True)
            print("keep_remainders=True")
            for _ in tqdm(sampler, total=len(sampler)):
                continue
            sampler = CustomBatchSamplerPillID(all_imgs_df, batch_size=batch_size, labelcol='label', min_classes=min_classes, min_per_class=min_per_class, keep_remainders=False, batch_size_mode=mode, debug=True)
            print("keep_remainders=False")
            for _ in tqdm(sampler, total=len(sampler)):
                continue

    # mask = all_imgs_df['label'].isin(ref_df['label'].unique())
    # min_per_class_range = np.linspace(2, all_imgs_df[mask].value_counts('label').unique().max(), 2, dtype=int)
    # for min_per_class in min_per_class_range:
    #     for min_per_class in min_per_class_range:
    #         val_counts = all_imgs_df[mask].value_counts('label')
    #         val_counts = val_counts[val_counts >= min_per_class]
    #         max_classes = len(val_counts)
    #         max_batch_size = all_imgs_df['label'].isin(val_counts.index.tolist()).sum()
    #         for min_classes in np.linspace(2, max_classes, 2, dtype=int):
    #             min_batch_size = min_classes*min_per_class
    #             for batch_size in np.linspace(min_batch_size, max_batch_size, 3, dtype=int):
    #                 sampler_args.append((min_per_class, min_classes, batch_size))

    # for mode in ['min', 'max', 'strict', None]:
    #     print("{}:".format(mode if mode is not None else 'None'))
    #     for min_per_class, min_classes, batch_size in sampler_args:
    #         print("min_per_class={}, min_classes={}, batch_size:{}".format(min_per_class, min_classes, batch_size))
    #         sampler = CustomBatchSamplerPillID(all_imgs_df, batch_size=batch_size, labelcol='label', min_classes=min_classes, min_per_class=min_per_class, refs_per_class=1, keep_remainders=True, batch_size_mode=mode, debug=True)
    #         print("keep_remainders=True")
    #         for _ in tqdm(sampler, total=len(sampler)):
    #             continue
    #         sampler = CustomBatchSamplerPillID(all_imgs_df, batch_size=batch_size, labelcol='label', min_classes=min_classes, min_per_class=min_per_class, refs_per_class=1, keep_remainders=False, batch_size_mode=mode, debug=True)
    #         print("keep_remainders=False")
    #         for _ in tqdm(sampler, total=len(sampler)):
    #             continue
    
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
    