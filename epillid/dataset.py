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
    def __init__(self, df, batch_size, labelcol="pilltype_id", min_per_class=2, min_classes=2, batch_size_mode=None, refs_per_class=0, generator=None):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        self.min_per_class = min_per_class # drops any classes that don't have at least min_per_class instances
        self.valid_classes = self.df[self.labelcol].value_counts().where(lambda x: x > 1).index.tolist() #dropping any labels with less than self.min_per_class instances
        # if refs_per_class > 0:
        #     if self.df[self.df.is_ref][self.labelcol].value_counts().max() < self.min_per_class:
        #         self.min_classes = self.df[self.df.is_ref][self.labelcol].value_counts().max() + 1
        #     self.valid_classes = np.intersect1d(self.df[self.df.is_ref][self.labelcol].unique(), self.valid_classes).tolist() #dropping labels with no ref images
        #     self.df = self.df[self.df[self.labelcol].isin(self.valid_classes)]
        #     self.refs = {k:v.values for k,v in self.df[self.df.is_ref].groupby(self.labelcol).groups}
        # else:
        #     self.refs = None
        #     self.df = self.df[self.df[self.labelcol].isin(self.valid_classes)]
        self.df = self.df[self.df[self.labelcol].isin(self.valid_classes)]
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        self.refs_per_class = refs_per_class
        if self.refs_per_class > 0:
            assert refs_per_class < self.min_per_class
            self.refs = {k: self.rng.choice(v.values, self.refs_per_class, replace=False) for k,v in self.df[self.df.is_ref].groupby(self.labelcol).groups if len(v) >= self.refs_per_class}
            self.valid_classes = np.intersect1d(list(self.refs.keys()), self.valid_classes)
            self.df = self.df[self.df[self.labelcol].isin(self.valid_classes)]
        else:
            self.refs_per_class = 0
            self.refs = None
        self.min_classes = min_classes # requires at least min_classes classes in each batch (may repeat towards end if not enough classes remaining)
        assert (self.min_classes * self.min_per_class) <= self.batch_size
        self.batch_size_mode = batch_size_mode
        '''
        max: batches will be at most self.batch_size, but may be smaller
        min: batches will be a minimum of self.batch_size, but may be larger
        strict: batches will be exactly self.batch_size, but may break min_per_class condition
        None: will try to make batches of size self.batch_size but may be more or less as needed
        '''
        assert batch_size_mode in ["max", "min", "strict", None]

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
        if unseen[label] == 0:
            unseen.pop(label)
        if label not in seen.keys():
            seen[label] = []
        seen[label].extend(inds)

    def get_min_size(self, label):
        return self.min_per_class if self.refs is None else (self.min_per_class - len(self.refs[label]))

    def add_to_curr_batch(self, curr_batch, curr_batch_labels, label, ind_dict):
        if label not in curr_batch_labels:
            if self.refs is not None:
                curr_batch.extend(self.refs[label])
            inds_to_add = self.rng.choice(ind_dict[label], self.min_per_class - self.refs_per_class, replace=False)
            curr_batch_labels.append(label)
        else:
            inds_to_add = self.rng.choice(ind_dict[label], 1, replace=False)
        curr_batch.extend(inds_to_add)
        return inds_to_add
        
        
    def __iter__(self):
        #maybe just shuffle once at beginning instead of always using rng.choice
        if self.refs is None:
            unseen = {k:v.values for k,v in self.df.groupby(self.labelcol).groups}
        else:
            unseen = {k:v.values for k,v in self.df[~self.df.is_ref].groupby(self.labelcol).groups}
        seen = {}
        
        choose_num_inds = self.min_per_class - self.refs_per_class
        while len(unseen) > 0:
            curr_batch = []
            curr_batch_labels = []
            leftovers = []
            while len(np.unique(curr_batch_labels)) < self.min_classes:
                # add new classes until min_classes in batch
                class_choices = np.setdiff1d(list(unseen.keys()), curr_batch_labels)
                add_seen = (len(class_choices) == 0)
                if add_seen:
                    #if no classes left with unseen images, add classes that were already seen and not in batch
                    curr_label = self.rng.choice(np.setdiff1d(list(self.seen_indicies.keys()),curr_batch_labels))
                    indicies = self.rng.choice(seen[curr_label], choose_num_inds, replace=False)
                else:
                    indicies = self.rng.choice(unseen[curr_label], choose_num_inds, replace=False)
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
                indicies = self.rng.choice(indicies, min(choose_num_inds, len(indicies)), replace=False)
                self.update_seen_unseen(seen, unseen, curr_label, indicies)
                
                if curr_label not in curr_batch_labels:
                    if self.refs is not None:
                        curr_batch.extend(self.refs[curr_label])
                        curr_batch_labels.append(curr_label)
                    
                curr_batch.extend(indicies)
                size_diff = self.batch_size - len(curr_batch)
                
            leftovers = [unseen.pop(l) for l in np.intersect1d(list(unseen.keys()), curr_batch_labels) if len(unseen[l]) < choose_num_inds]
            leftovers = np.array(leftovers).flatten()
            curr_batch.extend(leftovers)
            size_diff = self.batch_size - len(curr_batch)

            if size_diff > 0 and self.batch_size_mode in ['min', 'strict']:
                available = sum([np.intersect1d(seen[k], curr_batch) for k in curr_batch_labels], [])
                curr_batch.extend(self.rng.choice(available, min(size_diff, len(available)), replace=False))
                size_diff = self.batch_size - len(curr_batch)
                while size_diff > 0:
                    class_choices = np.setdiff1d(list(seen.keys()), curr_batch_labels)
                    curr_label = self.rng.choice(class_choices)
                    if self.refs is not None:
                        curr_batch.extend(self.refs[curr_label])
                        curr_batch_labels.append(curr_label)
                    indicies = seen[curr_label]
                    indicies = self.rng.choice(indicies, min(size_diff, len(indicies)), replace=False)
                    curr_batch.extend(indicies)
                    size_diff = self.batch_size - len(curr_batch)
            if size_diff < 0 and self.batch_size_mode in ['max', 'strict']:
                curr_batch = curr_batch[:size_diff]

            assert self.verify_batchsize(curr_batch)
            yield curr_batch

    def __len__(self):
        return len(self.df)//self.batch_size

if __name__ == "__main__":
    all_imgs_df, fold_indicies = utils.load_data()
    unique_classes = all_imgs_df['label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)
    ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    partitions = utils.split_data(all_imgs_df, fold_indicies)
    datasets = utils.get_datasets(partitions, ref_df, labelcol='label', two_sided=False, label_encoder = label_encoder)
    train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=32, labelcol='label', always_include_refs=True)
    val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=32, labelcol='label', always_include_refs=True)
    dataloaders = {'train': DataLoader(datasets['train'], batch_sampler=train_batch_sampler), 'val': DataLoader(datasets['val'], batch_sampler=val_batch_sampler)}

    for i,x in enumerate(tqdm(dataloaders['train'], total=len(dataloaders['train']))):
        continue