import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import BatchSampler
# import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
# import torchvision.transforms.v2 as transforms
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import euclidean_distances
# from pytorch_metric_learning import losses, miners, samplers, distances, trainers
import os
# import numpy as np
# import copy
from tqdm import tqdm
# import pandas as pd
# from PIL import Image

from model import ModelWrapper, EmbeddingModel
# import eval
import utils
from loss import TripletLoss, LossWrapper
# import metrics
from dataset import CustomBatchSamplerPillID

# class Trainer:
#     def __init__(self, device, model, dataloaders, clip_gradients, optimizers, lr_schedulers, criterion, two_sided, use_side_labels, use_ref, model_dir='./models', log_dir='./logs'):
#         self.device = device
#         self.model = model.to(device)
#         self.dataloaders = dataloaders
#         self.clip_gradients = clip_gradients
#         self.optimizers = optimizers
#         self.lr_schedulers = lr_schedulers
#         self.criterion = criterion
#         self.two_sided = two_sided
#         self.use_side_labels = use_side_labels 
#         self.use_ref = use_ref
#         self.train_loss_tracker = loss.LossTracker()
#         self.val_loss_tracker = loss.LossTracker()
#         self.model_dir = model_dir
#         self.log_dir = log_dir
#         os.makedirs(model_dir, exist_ok=True)
#         os.makedirs(log_dir, exist_ok=True)
#         self.writer = SummaryWriter(log_dir=self.log_dir)
        
#     def train_loop(self):
#         self.model.train()
#         for i,data in enumerate(tqdm(self.dataloaders["train"], total=len(self.dataloaders["train"]))):
#             labels = data[0].to(self.device)
#             imgs = data[1].to(self.device)
#             is_front = data[2].to(self.device) if (self.two_sided and self.use_side_labels) else None

#             if self.use_ref:
#                 is_ref = data[2].to(self.device) if self.two_sided else data[3].to(self.device)
#             else:
#                 is_ref = None

#             for opt in self.optimizers:
#                 opt.zero_grad()
                
#             with torch.set_grad_enabled(True):
#                 embeddings, logits = self.model(imgs)
                
#                 loss = self.criterion(embeddings, logits, labels, is_front, is_ref)
#                 loss['total'].backward()

#                 if self.clip_gradients:
#                     nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                
#                 for opt in self.optimizers:
#                     opt.step()
            
#             for k,v in loss.items():
#                 self.train_loss_tracker.update_curr_loss(k,v)
                    
#     def val_loop(self):
#         self.model.eval()
#         for i,data in enumerate(tqdm(self.dataloaders["val"], total=len(self.dataloaders["val"]))):
#             labels = data[0].to(self.device)
#             imgs = data[1].to(self.device)
#             is_front = data[2].to(self.device) if (self.two_sided and self.use_side_labels) else None

#             if self.use_ref:
#                 is_ref = data[2].to(self.device) if self.two_sided else data[3].to(self.device)
#             else:
#                 is_ref = None
#             with torch.set_grad_enabled(False):
#                 embeddings, logits = self.model(imgs)
#                 loss = self.criterion(embeddings, logits, labels, is_front, is_ref)
#             for k,v in loss.items():
#                 self.val_loss_tracker.update_curr_loss(k,v)


#     def train(self, num_epochs, checkpoint=3, sub_dir=None):
#         print("Beginning Training...")
#         save_path = self.model_dir if sub_dir is None else os.path.join(self.model_dir, sub_dir)
#         os.makedirs(save_path, exist_ok=True)
#         for i in range(num_epochs):
#             print("Training Epoch " + str(i) + "...")
#             self.train_loop()
#             train_loss = self.train_loss_tracker.update_loss_history()
#             for k,v in train_loss.items():
#                 self.writer.add_scalar('train_' + k, v, i)
#             print("Training complete, total train_loss was {:f}".format(train_loss['total']))
#             print("Starting validation loop...")
#             self.val_loop()
#             val_loss = self.val_loss_tracker.update_loss_history()
#             for k,v in val_loss.items():
#                 self.writer.add_scalar('val_' + k, v, i)
#             print("Validation loop complete")
#             if (i%checkpoint) == 0:
#                 filename = 'epoch_' + str(i)
#                 path = os.path.join(save_path, filename)
#                 torch.save(self.model, path)
#                 print("Saved to " + path)

#             for lr_scheduler in self.lr_schedulers:
#                 if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
#                     lr_scheduler.step(val_loss['total'])
#                 else:
#                     lr_scheduler.step()
#         self.writer.flush()
#         self.writer.close()
#         print("Logs saved to ", self.log_dir)
#         if ((num_epochs-1)%checkpoint) != 0:
#             filename = '_epoch_' + str(num_epochs-1)
#             path = os.path.join(save_path, filename)
#             torch.save(self.model, path)
#             print("Saved to " + path)

from loss import LossTracker
from eval import MetricTracker
from sklearn.metrics import top_k_accuracy_score
class Trainer:
    def __init__(self, n_classes, device, model, dataloaders, clip_gradients, optimizer, lr_scheduler, criterion, writer, use_ref_labels=True, use_side_labels=True, path="./"):
        self.n_classes = n_classes
        self.device = device
        self.class_inds = torch.arange(n_classes)
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.clip_gradients = clip_gradients
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.loss_trackers = {'train': LossTracker(), 'val': LossTracker()}
        self.metric_trackers = {'train': MetricTracker(), 'val': MetricTracker()}
        self.writer = writer
        self.use_ref_labels = use_ref_labels
        self.use_side_labels = use_side_labels
        self.path = path
        os.makedirs(os.path.join(self.path, "checkpoints"), exist_ok=True)

    def eval(self, labels, logits):
        avg_top_1_acc = top_k_accuracy_score(labels.detach().cpu(), logits.detach().cpu(), k=1, labels=self.class_inds)
        avg_top_5_acc = top_k_accuracy_score(labels.detach().cpu(), logits.detach().cpu(), k=5, labels=self.class_inds)
        return {"avg_top_1_acc": avg_top_1_acc, "top_5_acc": avg_top_5_acc}

    def train_loop(self):
        self.model.train()
        for data in tqdm(self.dataloaders["train"], total=len(self.dataloaders["train"])):
            labels = data[0].to(self.device).long()
            imgs = data[1].to(self.device)
            is_front = data[2].to(self.device) if self.use_side_labels else None
            is_ref = data[3].to(self.device) if self.use_ref_labels else None
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                embeddings, logits = self.model(imgs)
                loss = self.criterion({"embeddings": embeddings, "logits": logits, "labels":labels, "is_front": is_front, "is_ref": is_ref})
                loss["total"].backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
            self.loss_trackers['train'].update_curr_loss(loss)
            self.metric_trackers['train'].update_curr_metrics(self.eval(labels, logits))

    def val_loop(self):
        self.model.eval()
        for data in tqdm(self.dataloaders["val"], total=len(self.dataloaders["val"])):
            labels = data[0].to(self.device).long()
            imgs = data[1].to(self.device)
            is_front = data[2].to(self.device) if self.use_side_labels else None
            is_ref = data[3].to(self.device) if self.use_ref_labels else None
            with torch.set_grad_enabled(False):
                embeddings, logits = self.model(imgs)
                loss = self.criterion({"embeddings": embeddings, "logits": logits, "labels":labels, "is_front": is_front, "is_ref": is_ref})
            self.loss_trackers['val'].update_curr_loss(loss)
            self.metric_trackers['val'].update_curr_metrics(self.eval(labels, logits))

    def save_checkpoint(self, i):
        self.writer.flush()
        torch.save(self.model, os.path.join(self.path, "checkpoints", f'checkpoint_epoch_{i}.pth'))

    def train(self, num_epochs, checkpoint=3):
        for i in range(num_epochs):
            print(f"Epoch {i}")
            print("Running train loop...")
            self.train_loop()
            train_loss = self.loss_trackers['train'].update_loss_history()
            train_metrics = self.metric_trackers['train'].update_history()
            print("Running val loop...")
            self.val_loop()
            val_loss = self.loss_trackers['val'].update_loss_history()  
            val_metrics = self.metric_trackers['val'].update_history()     
            for k,v in train_loss.items():
                self.writer.add_scalars(k + '_loss', {'train': v, 'val': val_loss[k]}, i)
            for k,v in train_metrics.items():
                self.writer.add_scalars(k, {'train': v, 'val': val_metrics[k]}, i)  
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(val_loss['total'])
            else:
                self.lr_scheduler.step()
            if (i%checkpoint) == 0:
                self.save_checkpoint(i)
        if (i-1)%checkpoint != 0:
            self.save_checkpoint(i-1)
        self.writer.close()


if __name__ == "__main__":
    # all_imgs_df, fold_indicies = utils.load_data()
    # unique_classes = all_imgs_df['label'].unique()
    # label_encoder = LabelEncoder()
    # label_encoder.fit(unique_classes)
    # ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    # partitions = utils.split_data(all_imgs_df, fold_indicies)
    # datasets = utils.get_datasets(partitions, ref_df, labelcol='label', two_sided=False, label_encoder = label_encoder)
    # train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=32, labelcol='label', batch_size_mode='min')
    # val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=32, labelcol='label', batch_size_mode='min')
    # dataloaders = {'train': DataLoader(datasets['train'], batch_sampler=train_batch_sampler), 'val': DataLoader(datasets['val'], batch_sampler=val_batch_sampler)}
    # model = model.ModelWrapper(len(unique_classes),resnet50(weights=ResNet50_Weights.DEFAULT), False)
    
    # loss_types = {'embedding': {}, 'logit': {}}
    # loss_types['embedding']['triplet'] = loss.TripletLoss(miner=miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard"), triplet_loss=losses.TripletMarginLoss())
    # loss_types['logit']['cross_entropy'] = nn.CrossEntropyLoss()
    # loss_weights = {'triplet': 1.0, 'cross_entropy': 1.0}
    # criterion = loss.ModelLoss(len(unique_classes), loss_types, loss_weights, False, False)

    # optimizers = [optim.Adam(model.parameters(), lr=0.01)]
    # lr_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizers[0])]
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # trainer = Trainer(device, model, dataloaders, True, optimizers, lr_schedulers, criterion, two_sided=False, use_side_labels=False, use_ref=False, model_dir='./epillid/models', log_dir='./epillid/logs')
    # trainer.train(num_epochs=3, checkpoint=1)
    all_imgs_df, fold_indicies = utils.load_data()
    unique_classes = all_imgs_df['label'].unique()
    n_classes = len(unique_classes)
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)
    ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    partitions = utils.split_data(all_imgs_df, fold_indicies)
    datasets = utils.get_datasets(partitions, ref_df, 'label', False, label_encoder=label_encoder)
    dataloaders = {}
    for k,v in datasets.items():
        dataloaders[k] = DataLoader(v, batch_sampler=CustomBatchSamplerPillID(v.df, 32, labelcol='label', min_classes=5, min_per_class=3, keep_remainders=True, batch_size_mode='min', debug=False))

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(device)
    embedding_model = EmbeddingModel(resnet50(weights=ResNet50_Weights.DEFAULT)).to(device)
    writer = SummaryWriter("./training_logs")
    model = ModelWrapper(len(unique_classes), embedding_model)
    triplet_loss = TripletLoss(mode='hard')
    criterion = LossWrapper({'triplet': triplet_loss}, {'triplet': 1.0}, device=device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)
    trainer = Trainer(n_classes=n_classes, device=device, model=model, dataloaders=dataloaders, clip_gradients=True, optimizer=opt, lr_scheduler=lr_scheduler, criterion=criterion, writer=writer, path="./")
    trainer.train(10, checkpoint=5)