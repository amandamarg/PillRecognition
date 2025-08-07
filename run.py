from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.optim as optim
import torch.nn as nn

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
from sklearn.metrics import top_k_accuracy_score
import copy

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
        * mine triplets
        * get loss
        * if training, take optimizer step
    c) take scheduler step

'''

def load_epillid(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', path_to_folds = 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base', labelcol = 'label', train_with_side_labels = True, encode_labels = True):
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
        all_images_df[labelcol] = all_images_df.apply(lambda x: x[labelcol] + '_' + ('0' if x.is_front else '1'), axis=1)
        val_df[labelcol] = val_df.apply(lambda x: x[labelcol] + '_' + ('0' if x.is_front else '1'), axis=1)
        test_df[labelcol] = test_df.apply(lambda x: x[labelcol] + '_' + ('0' if x.is_front else '1'), axis=1)

    n_classes = all_images_df[labelcol].nunique()
    if encode_labels:
        label_encoder = LabelEncoder()
        encoded_label_name = 'encoded_' + labelcol
        all_images_df[encoded_label_name] = label_encoder.fit_transform(all_images_df[labelcol])
        val_df[encoded_label_name] = label_encoder.transform(val_df[labelcol])
        test_df[encoded_label_name] = label_encoder.transform(test_df[labelcol])

    val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
    train_df = all_images_df[~all_images_df['image_path'].isin(val_test_image_paths)]

    return {'train': train_df, 'val': val_df, 'test': test_df}, n_classes

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

def train_epoch(dataloader, device, models, miner, loss_funcs, optimizers, loss_dict, clip_gradients=True):
    embedding_losses = []
    classification_losses = []
    embedding_model.train()
    classifier.train()
    for data in tqdm(dataloader, total=len(dataloader)):
        imgs = data[0].to(device)
        labels = data[1].to(device)

        optimizers['embedding'].zero_grad()
        optimizers['classifier'].zero_grad()
        with torch.set_grad_enabled(True):
            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            embedding_loss = loss_funcs['embedding'](embeddings, labels, mined_output)
            embedding_losses.append(embedding_loss)

            logits = models['classifier'](embeddings)
            classification_loss = loss_funcs['classifier'](logits, labels)
            classification_losses.append(classification_loss)
            #TODO: compute metrics
            embedding_loss.loss_backward()
            classification_loss.backward()

            if clip_gradients:
                nn.utils.clip_grad_norm_(models['embedding'].parameters(), 1.)
                nn.utils.clip_grad_norm_(models['classifier'].parameters(), 1.)
                
            optimizers['embedding'].step()
            optimizers['classifier'].step()
    loss_dict['embedding'].append(np.average(embedding_losses))
    loss_dict['classifier'].append(np.average(classification_losses))
    return

def val_epoch(dataloader, device, models, miner, loss_funcs, loss_dict):
    embedding_losses = []
    classification_losses = []
    embedding_model.eval()
    classifier.eval()
    for data in tqdm(dataloader, total=len(dataloader)):
        imgs = data[0].to(device)
        labels = data[1].to(device)
        with torch.set_grad_enabled(False):
            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            embedding_loss = loss_funcs['embedding'](embeddings, labels, mined_output)
            embedding_losses.append(embedding_loss)

            logits = models['classifier'](embeddings)
            classification_loss = loss_funcs['classifier'](logits, labels)
            classification_losses.append(classification_loss)
            #TODO: compute metrics
    loss_dict['embedding'].append(np.average(embedding_losses))
    loss_dict['classifier'].append(np.average(classification_losses))
    return

def train(models, dataloaders, num_epochs, device,  miner, loss_funcs, optimizers, lr_schedulers):
    models["embedding"].to(device)
    models["classifier"].to(device)

    train_loss_avgs = {"embedding": [], "classifier": []}
    val_loss_avgs = {"embedding": [], "classifier": []}

    for i in range(num_epochs):
        print("Training Epoch {:d}...".format(i))
        train_epoch(dataloaders["train"], device, models, miner, loss_funcs, optimizers, train_loss_avgs)
        print("Training loss: embedding_model_loss={:f}, classifier_loss={:f}".format(train_loss_avgs["embedding"], train_loss_avgs["classifier"]))
        val_epoch(dataloaders["val"], device, models, miner, loss_funcs, val_loss_avgs)
        print("Validation loss: embedding_model_loss={:f}, classifier_loss={:f}".format(val_loss_avgs["embedding"], val_loss_avgs["classifier"]))
        lr_schedulers["embedding"].step(val_loss_avgs["embedding"])
        lr_schedulers["classifier"].step(val_loss_avgs["classifier"])
    
class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_size))
    
    def forward(self, input):
        distances = torch.norm(input, 2, dim=1, keepdim=True)
        return self.fc(torch.div(input,distances))

if __name__ == "__main__":
    df_dataset, n_classes = load_epillid()

    dataset_dict = {k : PillImages(v, k, labelcol='encoded_label') for k,v in df_dataset.items()}
    num_epochs = 1
    batch_size = 32
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    train_batch_sampler = CustomBatchSamplerPillID(dataset_dict['train'].df, batch_size=batch_size, labelcol='encoded_label')
    val_batch_sampler = CustomBatchSamplerPillID(dataset_dict['val'].df, batch_size=batch_size, labelcol='encoded_label')

    train_dataloader = DataLoader(dataset_dict['train'], batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(dataset_dict['val'], batch_sampler=val_batch_sampler)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    #initalize embedding model
    embedding_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    embedding_size = embedding_model.fc.in_features
    #peel off last layer
    embedding_model.fc = torch.nn.Identity()

    classifier = Classifier(embedding_size, n_classes)

    models = {"embedding": embedding_model, "classifier": classifier}

    loss_funcs = {"embedding": losses.TripletMarginLoss(), "classifier": nn.CrossEntropyLoss()}

    optimizers = {"embedding": optim.Adam(embedding_model.parameters(), lr=0.01), "classifier": optim.Adam(classifier.parameters(), lr=0.01)}

    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

    lr_schedulers = {"embedding": optim.lr_scheduler.ReduceLROnPlateau(optimizers["embedding"], factor=0.5, patience=2), "classifier": optim.lr_scheduler.ReduceLROnPlateau(optimizers["classifier"], factor=0.5, patience=2)}

    train(models, dataloaders, num_epochs, batch_size, device,  miner, loss_funcs, optimizers, lr_schedulers)

    # model_weights = copy.deepcopy(model.state_dict())


    
