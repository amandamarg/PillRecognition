from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd

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
        return {"label": label, "img": img, "is_front": int(row.is_front), "is_ref": int(row.is_ref)}
    
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
        return {"label": label, "img": (front_img, back_img), "is_ref": (front_row.is_ref, back_row.is_ref)}
    
    def load_img(self, img_path):
        if not os.path.exists(img_path):
            print("img not found", img_path)
            return
        to_tensor = transforms.Compose([transforms.PILToTensor(), transforms.ToDtype(torch.float32, scale=True)])
        return to_tensor(Image.open(img_path))
    
class CustomBatchSamplerPillID(BatchSampler):
    def __init__(self, df, batch_size, labelcol="pilltype_id", generator=None, min_per_class=2, add_refs=False, min_classes=2, batch_size_mode=None, keep_remainders=False, debug=False):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        self.min_per_class = min_per_class # drops any classes that don't have at least min_per_class instances
        val_counts = self.df.value_counts(self.labelcol)
        valid_classes = val_counts[val_counts >= self.min_per_class].index.values
        if add_refs:
            ref_val_counts = self.df[self.df.is_ref].value_counts(self.labelcol)
            assert ref_val_counts.max() == ref_val_counts.min() #make sure all classes have same number of reference images
            self.num_refs = ref_val_counts.max()
            assert self.num_refs < self.min_per_class
            valid_classes = np.intersect1d(self.df[self.df.is_ref][self.labelcol].unique(), valid_classes)
            self.refs = {k: v.values for k,v in self.df[self.df.is_ref].groupby(self.labelcol).groups.items() if k in valid_classes}
        else:
            self.num_refs = None
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

    def select(self, n, options, backups=None, shuffle=False):
        if shuffle:
            self.rng.shuffle(options)
        if len(options) < n and backups is not None:
            if shuffle:
                self.rng.shuffle(backups)
            return (options, backups[:min(len(backups), n-len(options))])
        if backups is not None:
            return (options[:n], [])
        return options[:min(n, len(options))]
        
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
                inds = self.select(size_diff, v)
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
        num_per_class = self.min_per_class if self.refs is None else self.min_per_class - self.num_refs
        for label in present_labels:
            if add_from_seen:
                inds = np.setdiff1d(seen[label], curr_batch)
            else:
                inds = unseen[label]
            selected_inds = self.select(size_diff, inds)
            if not add_from_seen:
                remaining = self.update_seen_unseen(seen, unseen, label, selected_inds)
                if len(remaining) < num_per_class and len(remaining) > 0:
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
            
        default = list(np.setdiff1d(list(seen.keys()), batch_labels)) if seen_is_default else list(np.setdiff1d(list(unseen.keys()), batch_labels)) 
        if not default_only:
            backups = list(unseen.keys()) if seen_is_default else list(seen.keys())
            backups = list(np.setdiff1d(backups, default + batch_labels))
            add_classes = np.concatenate(self.select(num_classes, default, backups, shuffle=True))
        else:
            add_classes = self.select(num_classes, default, shuffle=True)

        if len(add_classes) == 0:
            return 0
        
        leftovers = {}
        num_per_class = self.min_per_class if self.refs is None else self.min_per_class - self.num_refs
        for label in add_classes:
            assert label not in batch_labels
            if self.refs is not None:
                curr_batch.extend(self.refs[label])
            options = unseen[label] if label in unseen.keys() else []
            backups = seen[label] if label in seen.keys() else []
            selected = self.select(num_per_class, options, backups)
            if update_seen_unseen and len(selected[0]) > 0:
                remaining = self.update_seen_unseen(seen, unseen, label, selected[0])
                if len(remaining) < num_per_class and len(remaining) > 0:
                    leftovers[label] = remaining
            selected = np.concatenate(selected).astype(int)
            assert len(selected) == num_per_class
            curr_batch.extend(selected)
            batch_labels.append(label)

        if len(leftovers) > 0:
            self.cleanup_leftovers(leftovers, seen, unseen, curr_batch)
        
        return len(add_classes)
    
            
    def __iter__(self):
        if self.refs is None:
            unseen = {k: self.rng.choice(v.values, len(v), replace=False) for k,v in self.df.groupby(self.labelcol).groups.items() if k in self.valid_classes}
        else:
            unseen = {k: self.rng.choice(v.values, len(v), replace=False) for k,v in self.df[~self.df.is_ref].groupby(self.labelcol).groups.items() if k in self.valid_classes}
        
        seen = {}
        
        while len(unseen) > 0:
            curr_batch = []
            curr_batch_labels = []
            self.grow_new_classes(curr_batch, curr_batch_labels, unseen, seen, seen_is_default=False, default_only=False, num_classes=self.min_classes) # make sure we reach minimum number of classes by first adding from unseen, and then from seen only if needed
            # self.grow_existing_classes(curr_batch, curr_batch_labels, unseen, seen, add_from_seen=False)
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