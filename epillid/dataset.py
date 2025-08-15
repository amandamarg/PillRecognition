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

        return img, label
    
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
        labels = list(self.df[self.labelcol].unique())
        label_map= {k: self.df[self.df[self.labelcol] == k].index.tolist() for k in labels}
        while len(labels) > 0:
            curr_batch = []
            while self.batch_size > len(curr_batch):
                self.rng.shuffle(labels)
                curr_label = labels[0]
                indicies = label_map[curr_label]
                self.rng.shuffle(indicies)                    
                if len(indicies) < 4:
                    if len(indicies) > 1: # TODO: find a better way to deal with classses with only 1 image per label, but for now, we'll just ignore those classes
                        curr_batch.extend(indicies)
                        label_map[curr_label] = []
                    labels.remove(curr_label)
                    if len(labels) == 0:
                        break
                else:
                    curr_batch.extend(indicies[:2])
                    label_map[curr_label] = indicies[2:]
            yield curr_batch

    def __len__(self):
        return len(self.df)//self.batch_size
    