import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.v2 as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from pytorch_metric_learning import losses, miners, samplers, distances, trainers
import os
import numpy as np
import copy
from tqdm import tqdm
import pandas as pd
from PIL import Image

import model
import eval
import utils
import loss
import metrics
from dataset import CustomBatchSamplerPillID

class Trainer:
    def __init__(self, device, model, dataloaders, clip_gradients, optimizers, lr_schedulers, criterion, two_sided, use_side_labels, use_ref, train_metrics_tracker, val_metrics_tracker):
        self.device = device
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.clip_gradients = clip_gradients
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.criterion = criterion
        self.two_sided = two_sided
        self.use_side_labels = use_side_labels 
        self.use_ref = use_ref
        self.train_loss_tracker = loss.LossTracker()
        self.val_loss_tracker = loss.LossTracker()
        self.train_metrics_tracker = train_metrics_tracker
        self.val_metrics_tracker = val_metrics_tracker
        self.writer = SummaryWriter()
        
    def train_loop(self):
        self.model.train()
        for i,data in enumerate(tqdm(self.dataloaders["train"], total=len(self.dataloaders["train"]))):
            labels = data[0].to(self.device)
            imgs = data[1].to(self.device)
            is_front = data[2].to(self.device) if (self.two_sided and self.use_side_labels) else None

            if self.use_ref:
                is_ref = data[2].to(self.device) if self.two_sided else data[3].to(self.device)
            else:
                is_ref = None

            for opt in self.optimizers:
                opt.zero_grad()
                
            with torch.set_grad_enabled(True):
                embeddings, logits = self.model(imgs)
                
                loss = self.criterion(embeddings, logits, labels, is_front, is_ref)
                loss['total'].backward()

                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                
                for opt in self.optimizers:
                    opt.step()
            
            for k,v in loss.items():
                self.train_loss_tracker.update_curr_loss(k,v)

            self.train_metrics_tracker.update_batch(embeddings, logits, labels, is_ref, is_front)
                    
    def val_loop(self):
        self.model.eval()
        for i,data in enumerate(tqdm(self.dataloaders["val"], total=len(self.dataloaders["val"]))):
            labels = data[0].to(self.device)
            imgs = data[1].to(self.device)
            is_front = data[2].to(self.device) if (self.two_sided and self.use_side_labels) else None

            if self.use_ref:
                is_ref = data[2].to(self.device) if self.two_sided else data[3].to(self.device)
            else:
                is_ref = None
            with torch.set_grad_enabled(False):
                embeddings, logits = self.model(imgs)
                loss = self.criterion(embeddings, logits, labels, is_front, is_ref)
            for k,v in loss.items():
                self.val_loss_tracker.update_curr_loss(k,v)

            self.val_metrics_tracker.update_batch(embeddings, logits, labels, is_ref, is_front)


    def train(self, model_name, num_epochs, checkpoint=3, save_dir='/Users/Amanda/Desktop/PillRecognition/model'):
        for i in range(num_epochs):
            self.train_loop()
            train_loss = self.train_loss_tracker.update_loss_history()
            for k,v in train_loss.items():
                self.writer.add_scalar('train_' + k, v, i)
            train_metrics = self.train_metrics_tracker.update_metrics()
            for k,v in train_metrics.items():
                if isinstance(v, float):
                    self.writer.add_scalar('train_' + k, v, i)
                else:
                    self.writer.add_scalars('train_' + k, dict(zip(np.arange(len(v)), v)), i)

            self.val_loop()
            val_loss = self.val_loss_tracker.update_loss_history()
            for k,v in val_loss.items():
                self.writer.add_scalar('val_' + k, v, i)
            val_metrics = self.val_metrics_tracker.update_metrics()
            for k,v in val_metrics.items():
                if isinstance(v, float):
                    self.writer.add_scalar('val_' + k, v, i)
                else:
                    self.writer.add_scalars('val_' + k, dict(zip(np.arange(len(v)), v)), i)

            if (i%checkpoint) == 0:
                filename = model_name + '_epoch_' + str(i)
                path = os.path.join(save_dir, filename)
                torch.save(self.model, path)
                print("Saved to " + path)

            for lr_scheduler in self.lr_schedulers:
                if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    lr_scheduler.step(val_loss['total'])
                else:
                    lr_scheduler.step()
        self.writer.flush()
        self.writer.close()
        if ((num_epochs-1)%checkpoint) != 0:
            filename = model_name + '_epoch_' + str(num_epochs-1)
            path = os.path.join(save_dir, filename)
            torch.save(self.model, path)
            print("Saved to " + path)

if __name__ == "__main__":
    all_imgs_df, fold_indicies = utils.load_data()
    unique_classes = all_imgs_df['label'].unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)
    ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    partitions = utils.split_data(all_imgs_df, fold_indicies)
    datasets = utils.get_datasets(partitions, ref_df, labelcol='label', two_sided=False, label_encoder = label_encoder)
    train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=32, labelcol='label')
    val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=32, labelcol='label')
    dataloaders = {'train': DataLoader(datasets['train'], batch_sampler=train_batch_sampler), 'val': DataLoader(datasets['val'], batch_sampler=val_batch_sampler)}
    model = model.ModelWrapper(len(unique_classes),resnet50(weights=ResNet50_Weights.DEFAULT), False)
    
    loss_types = {'embedding': {}, 'logit': {}}
    loss_types['embedding']['triplet'] = loss.TripletLoss(miner=miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard"), triplet_loss=losses.TripletMarginLoss())
    loss_types['logit']['cross_entropy'] = nn.CrossEntropyLoss()
    loss_weights = {'triplet': 1.0, 'cross_entropy': 1.0}
    criterion = loss.ModelLoss(len(unique_classes), loss_types, loss_weights, False, False)

    optimizers = [optim.Adam(model.parameters(), lr=0.01)]
    lr_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizers[0])]
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    train_metric_tracker = metrics.MetricTracker({'map_5':metrics.MAP_K(5,use_min=True),'mrr': metrics.MRR()}, {'map_5': metrics.MAP_K(5,use_min=True), 'mrr': metrics.MRR()}, 'train')
    val_metric_tracker = metrics.MetricTracker({'map_5':metrics.MAP_K(5,use_min=True),'mrr': metrics.MRR()}, {'map_5': metrics.MAP_K(5,use_min=True), 'mrr': metrics.MRR()}, 'val')

    trainer = Trainer(device, model, dataloaders, True, optimizers, lr_schedulers, criterion, two_sided=False, use_side_labels=False, use_ref=False, train_metrics_tracker=train_metric_tracker, val_metrics_tracker=val_metric_tracker)
    trainer.train("model_test", 3, 1)

