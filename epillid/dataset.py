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
    def __init__(self, front_df, back_df, transform=None, augment=None, labelcol="pilltype_id", label_encoder=None):
        assert len(front_df) == len(back_df)
        self.front_df = front_df
        self.back_df = back_df
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
    def __init__(self, df, batch_size, labelcol="pilltype_id", generator=None):
        self.df = df.copy().reset_index() # the dataset uses .iloc
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
            
    def __len__(self):
        return int(self.df[self.labelcol].value_counts().where(lambda x: x > 1).sum())//self.batch_size

