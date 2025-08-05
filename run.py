from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.optim as optim

import os
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image
from pytorch_metric_learning import losses, miners, samplers, distances

from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import LabelEncoder


'''
Steps:

1) Read in all data
2) Encode labels
    * need to decide whether to treat front and back images as single class or seperate class and split accordingly
3) Split data
    * training data should have at least two images per class per side so that we can generate triplet
    * should training have all reference images for each class?
4) Load splits into dataset
5) For each epoch
    a) generate mini-batches
        * for each batch, we need to make sure that there are valid triplets in the batch, which we do using method in https://challengeenthusiast.com/training-a-siamese-model-with-a-triplet-loss-function-on-mnist-dataset-using-pytorch-225908e59bda
        * basically, we generate pairs of same-class images and then each mini batch has only one pair per class/subset of classes
    b) iterate through mini-batches
        * get embeddings
        * 

'''


def load_epillid(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', path_to_folds = 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base', train_with_side_labels = True, encode_labels=True):
    data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata'
    csv_files = glob(os.path.join(data_root_dir, path_to_folds, '*.csv'))

    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    csv_files = sorted([x for x in csv_files if not x.endswith("all.csv")])
    test_imgs_csv = csv_files.pop(-1)
    val_imgs_csv = csv_files.pop(-1)

    all_images_df = pd.read_csv(all_imgs_csv)
    val_df = pd.read_csv(val_imgs_csv)
    test_df = pd.read_csv(test_imgs_csv)

    img_dir = 'classification_data'
    for df in [all_images_df, val_df, test_df]:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(data_root_dir, img_dir, x))

    if train_with_side_labels:
        all_images_df['label'] = all_images_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)
        val_df['label'] = val_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)
        test_df['label'] = test_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)

    if encode_labels:
        label_encoder = LabelEncoder().fit(all_images_df.label)
        all_images_df['encoded_label'] = label_encoder.transform(all_images_df.label)
        val_df['encoded_label'] = label_encoder.transform(val_df.label)
        test_df['encoded_label'] = label_encoder.transform(test_df.label)

    val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
    train_df = all_images_df[~all_images_df['image_path'].isin(val_test_image_paths)]

    return {'train': train_df, 'val': val_df, 'test': test_df}


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


def train_epoch(dataloader, device, model, miner, loss, optimizer):
    losses = []
    model = model.to(device)
    model.train()
    for data in tqdm(dataloader, total=len(dataloader)):
        imgs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            embeddings = model(imgs)
            mined_output = miner(embeddings, labels)
            curr_loss = loss(embeddings, labels, mined_output)
            losses.append(curr_loss.item())
            #TODO: compute metrics
            curr_loss.backward()
            optimizer.step()
    return np.array(losses)

def val_epoch(dataloader, device, model, miner, loss):
    losses = []
    model = model.to(device)
    model.eval()
    for data in tqdm(dataloader, total=len(dataloader)):
        imgs = data[0].to(device)
        labels = data[1].to(device)
        with torch.set_grad_enabled(False):
            embeddings = model(imgs)
            mined_output = miner(embeddings, labels)
            curr_loss = loss(embeddings, labels, mined_output)
            losses.append(curr_loss.item())
    
    return np.array(losses)
    

if __name__ == "__main__":
    df_dataset = load_epillid()

    dataset_dict = {k : PillImages(v, k, labelcol='encoded_label') for k,v in df_dataset.items()}
    num_epochs = 3
    batch_size = 32
    device = torch.device('mps')

    train_batch_sampler = CustomBatchSamplerPillID(dataset_dict['train'].df, batch_size=batch_size, labelcol='encoded_label')
    val_batch_sampler = CustomBatchSamplerPillID(dataset_dict['val'].df, batch_size=batch_size, labelcol='encoded_label')

    train_dataloader = DataLoader(dataset_dict['train'], batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(dataset_dict['val'], batch_sampler=val_batch_sampler)

    #initalize model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    #peel off last layer
    model.fc = torch.nn.Identity()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    loss = losses.TripletMarginLoss()

    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

    train_loss_avgs = []
    val_loss_avgs = []
    for i in range(num_epochs):
        train_loss = train_epoch(train_dataloader, device, model, miner, loss, optimizer)
        val_loss = val_epoch(val_dataloader, device, model, miner, loss)
        train_loss_avgs.append(np.average(train_loss))
        val_loss_avgs.append(np.average(val_loss))
        print("Train loss: ", train_loss_avgs[-1])
        print("Val loss: ", val_loss_avgs[-1])
        scheduler.step(val_loss_avgs[-1])