from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
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
    def __init__(self, df, batch_size, labelcol="pilltype_id", min_per_class=2, min_classes=2, batch_size_mode=None, generator=None):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        self.min_per_class = min_per_class # drops any classes that don't have at least min_per_class instances
        self.min_classes = min_classes # requires at least min_classes different classes in each minibatch (may repeat towards end if not enough classes remaining)
        assert (self.min_classes * self.min_per_class) <= self.batch_size
        self.batch_size_mode = batch_size_mode
        '''
        max: batches will be at most self.batch_size, but may be smaller
        min: batches will be a minimum of self.batch_size, but may be larger
        strict: batches will be exactly self.batch_size, but may break min_per_class
        None: will try to make batches of size self.batch_size but may be more or less as needed
        '''
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()
        assert batch_size_mode in ["max", "min", "strict", None]

    # def batch_size_condition(self, curr_batch_size):
    #     if self.batch_size_mode == "max":
    #         return curr_batch_size + self.min_per_class


    def __iter__(self):
        unused_indicies = {k: v.values for k,v in self.df.groupby(self.labelcol).groups.items() if len(v) >= self.min_per_class} #dropping any labels with less than self.min_per_class instances
        valid_classes = set(list(unused_indicies.keys()))


        while len(unused_indicies) > 0:
            curr_batch = []
            curr_batch_labels = set()
            leftovers = []
            size_diff = self.batch_size
            while size_diff >= self.min_per_class:
                class_choices = list(set(unused_indicies.keys()) - curr_batch_labels) if len(curr_batch_labels) < self.min_classes else list(unused_indicies.keys())
                if len(class_choices) == 0:
                    curr_label = self.rng.choice(valid_classes - curr_batch_labels)
                    curr_batch.extend(self.df[self.df[self.labelcol] == curr_label].index.tolist())
                    curr_batch_labels.add(curr_label)
                else:
                    curr_label = self.rng.choice(class_choices)
                    indicies = self.rng.choice(unused_indicies[curr_label], self.min_per_class, replace=False)
                    curr_batch_labels.add(curr_label)
                    for i in indicies:
                        curr_batch.append(i)
                        unused_indicies[curr_label].remove(i)
                    if len(unused_indicies[curr_label]) < self.min_per_class:
                        leftovers.extend(unused_indicies.pop(curr_label))
                size_diff = self.batch_size - len(curr_batch)

            if self.batch_size_mode in [None, 'min']:
                curr_batch.extend(leftovers)
            else:
                curr_batch.extend(self.rng.choice(leftovers, min(len(leftovers), size_diff), replace=False))
            
            if (len(curr_batch) < self.batch_size) and self.batch_size_mode in ['min', 'strict']:
                curr_label = self.rng.choice(valid_classes - curr_batch_labels)
                curr_batch.extend(self.df[self.df[self.labelcol] == curr_label].index.tolist())
                if self.batch_size_mode == 'strict':
                    curr_batch = curr_batch[:self.batch_size]

            yield curr_batch


class ConsRefPairSampler(BatchSampler):
    def __init__(self, df, batch_size, labelcol="pilltype_id", generator=None, batch_size_mode="strict"):
        self.ref_df = df[df.is_ref].copy().reset_index() # the dataset uses .iloc
        self.cons_df = df[~df.is_ref].copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.labelcol = labelcol
        if generator:
            self.rng = generator
        else:
            self.rng = np.random.default_rng()

    def __iter__(self):
        label_map = {k: v.values for k,v in self.df.groupby(self.labelcol).groups.items() if len(v) > 1} # dropping any labels with only 1 index per label
        while len(label_map) > 0:
            curr_batch = []
            while self.batch_size > len(curr_batch):
                curr_label = self.rng.choice(list(label_map.keys()))
                indicies = label_map[curr_label]
                if len(indicies) < 4: # if 3 or less labels, just add all of them and then remove from label map
                    curr_batch.extend(indicies)
                    label_map.pop(curr_label)
                    if len(label_map) == 0:
                        break
                else:
                    selected_indicies = self.rng.choice(indicies, 2, replace=False)
                    curr_batch.extend(selected_indicies)
                    label_map[curr_label] = indicies[~np.isin(indicies, selected_indicies)]
            assert (len(curr_batch) > 0)
            yield curr_batch
