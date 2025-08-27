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

def train_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, optimizers, loss_dict, epoch_embedding_metrics, epoch_logit_metrics, clip_gradients=True):
    
    models["embedding"].train()
    models["classifier"].train()

    running_loss = {'embedding': [], 'classifier': [], 'total': []}
    embedding_metrics = {'map_1': [], 'map_5': [], 'MRR': []}
    logit_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}

    with torch.set_grad_enabled(True):
        for i,data in enumerate(tqdm(dataloader, total=len(dataloader))):
            labels = data[0].to(device)
            imgs = data[1].to(device)
            curr_loss = {'embedding': 0.0, 'classifier': 0.0}

            optimizers['embedding'].zero_grad()
            optimizers['classifier'].zero_grad()

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            curr_loss['embedding'] = loss_funcs['embedding'](embeddings, labels, mined_output)
            
            logits = models['classifier'](embeddings)
            curr_loss['classifier'] = loss_funcs['classifier'](logits, labels)

            batch_embedding_metrics = eval.eval_embeddings(labels.cpu().detach(), embeddings.cpu().detach())
            for k,v in batch_embedding_metrics.items():
                embedding_metrics[k].append(v)

            batch_logit_metrics = eval.eval_logits(labels.cpu().detach(), logits.cpu().detach())
            for k,v in batch_logit_metrics.items():
                logit_metrics[k].append(v)
                
            total_loss = 0.0
            for k,v in loss_weights.items():
                total_loss += (curr_loss[k] * v)
            total_loss.backward()

            running_loss['embedding'].append(curr_loss['embedding'].item())
            running_loss['classifier'].append(curr_loss['classifier'].item())
            running_loss['total'].append(total_loss.item())

            if clip_gradients:
                nn.utils.clip_grad_norm_(models['embedding'].parameters(), 1.)
                nn.utils.clip_grad_norm_(models['classifier'].parameters(), 1.)
                
            optimizers['embedding'].step()
            optimizers['classifier'].step()


    for k, v in running_loss.items():
        loss_dict[k].append(np.array(v)/len(v))
        
    for k, v in embedding_metrics.items():
        epoch_embedding_metrics[k].append(np.array(v)/len(v))
    
    for k, v in logit_metrics.items():
        epoch_logit_metrics[k].append(np.array(v)/len(v))

    return running_loss, embedding_metrics, logit_metrics

def val_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, loss_dict, epoch_embedding_metrics, epoch_logit_metrics):
    models["embedding"].eval()
    models["classifier"].eval()

    running_loss = {'embedding': [], 'classifier': [], 'total': []}
    embedding_metrics = {'map_1': [], 'map_5': [], 'MRR': []}
    logit_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}


    with torch.set_grad_enabled(False):
        for i,data in enumerate(tqdm(dataloader, total=len(dataloader))):
            curr_loss = {'embedding': 0.0, 'classifier': 0.0}

            labels = data[0].to(device)
            imgs = data[1].to(device)
            

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            curr_loss['embedding'] = loss_funcs['embedding'](embeddings, labels, mined_output)

            logits = models['classifier'](embeddings)
            curr_loss['classifier'] = loss_funcs['classifier'](logits, labels)

            batch_embedding_metrics = eval.eval_embeddings(labels.cpu().detach(), embeddings.cpu().detach())
            for k,v in batch_embedding_metrics.items():
                embedding_metrics[k].append(v)

            batch_logit_metrics = eval.eval_logits(labels.cpu().detach(), logits.cpu().detach())
            for k,v in batch_logit_metrics.items():
                logit_metrics[k].append(v)

            total_loss = 0.0
            for k,v in loss_weights.items():
                total_loss += (curr_loss[k] * v)

            running_loss['embedding'].append(curr_loss['embedding'].item())
            running_loss['classifier'].append(curr_loss['classifier'].item())
            running_loss['total'].append(total_loss.item())

    for k, v in running_loss.items():
        loss_dict[k].append(np.array(v)/len(v))
        
    for k, v in embedding_metrics.items():
        epoch_embedding_metrics[k].append(np.array(v)/len(v))
    
    for k, v in logit_metrics.items():
        epoch_logit_metrics[k].append(np.array(v)/len(v))

    return running_loss, embedding_metrics, logit_metrics

def train(model_name, models, dataloaders, num_epochs, device,  miner, loss_funcs, loss_weights, optimizers, lr_scheduler):
    models["embedding"].to(device)
    models["classifier"].to(device)

    train_loss_avgs = {"embedding": [], "classifier": [], "total": []}
    val_loss_avgs = {"embedding": [], "classifier": [], "total": []}
    train_epoch_embedding_metrics = {'map_1': [], 'map_5': [], 'MRR': []}
    train_epoch_logits_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}
    val_epoch_embedding_metrics = {'map_1': [], 'map_5': [], 'MRR': []}
    val_epoch_logits_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}


    for i in range(num_epochs):
        print("Training Epoch {:d}...".format(i))
        train_epoch(dataloaders["train"], device, models, miner, loss_funcs, loss_weights, optimizers, train_loss_avgs, train_epoch_embedding_metrics, train_epoch_logits_metrics, True)
        print("Training loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(train_loss_avgs["embedding"][0][-1], train_loss_avgs["classifier"][0][-1], train_loss_avgs["total"][0][-1]))
        print("Training Embedding Metrics:")
        for metric, metric_val in train_epoch_embedding_metrics.items():
            print(metric + '={:f}'.format(metric_val[0][-1]))
        print("Training Logits Metrics:")
        for metric, metric_val in train_epoch_logits_metrics.items():
            print(metric + '={:f}'.format(metric_val[0][-1]))
        val_epoch(dataloaders["val"], device, models, miner, loss_funcs, loss_weights, val_loss_avgs, val_epoch_embedding_metrics, val_epoch_logits_metrics)
        print("Validation loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(val_loss_avgs["embedding"][0][-1], val_loss_avgs["classifier"][0][-1], val_loss_avgs["total"][0][-1]))
        print("Validation Embedding Metrics:")
        for metric, metric_val in val_epoch_embedding_metrics.items():
            print(metric + '={:f}'.format(metric_val[0][-1]))
        print("Validation Logits Metrics:")
        for metric, metric_val in val_epoch_logits_metrics.items():
            print(metric + '={:f}'.format(metric_val[0][-1]))
        lr_scheduler["embedding"].step(val_loss_avgs["embedding"][0][-1])
        lr_scheduler["classifier"].step(val_loss_avgs["classifier"][0][-1])
        utils.save_model(models, model_name, i)
    return {"loss": train_loss_avgs, "embedding_metrics": val_epoch_embedding_metrics, "logits_metrics": train_epoch_logits_metrics}, {"loss": val_loss_avgs, "embedding_metrics": train_epoch_embedding_metrics, "logits_metrics": val_epoch_logits_metrics}


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
                
                loss = self.criterion(embeddings, labels, is_front, is_ref)
                loss['total'].backward()

                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                
                for opt in self.optimizers:
                    opt.step()
            
            for k,v in loss:
                self.train_loss_tracker.update_curr_loss(k,v)

            self.train_metrics_tracker.update_batch(embeddings, logits, labels, is_ref, is_front)
                    
    def val_loop(self):
        self.model.eval()
        for i,data in enumerate(tqdm(self.dataloaders["val"], total=len(self.dataloaders["val"]))):
            labels =data[0].to(self.device)
            imgs = data[1].to(self.device)
            is_front = data[2].to(self.device) if (self.two_sided and self.use_side_labels) else None

            if self.use_ref:
                is_ref = data[2].to(self.device) if self.two_sided else data[3].to(self.device)
            else:
                is_ref = None
            with torch.set_grad_enabled(True):
                embeddings, logits = self.model(imgs)
                loss = self.loss_func(embeddings, labels, is_front, is_ref)
            for k,v in loss:
                self.val_loss_tracker.update_curr_loss(k,v)

            self.val_metrics_tracker.update_batch(embeddings, logits, labels, is_ref, is_front)


    def train(self, model_name, num_epochs, checkpoint=3, save_dir='/Users/Amanda/Desktop/PillRecognition/model'):
        for i in range(num_epochs):
            self.train_loop()
            train_loss = self.train_loss_tracker.update_loss_history()
            for k,v in train_loss:
                self.writer.add_scalar('train_' + k, v, i)
            train_metrics = self.train_metrics_tracker.update_metrics()
            for k,v in train_metrics:
                if len(v) > 1:
                    self.writer.add_scalars('train_' + k, dict(zip(np.arange(len(v)), v)), i)
                else:
                    self.writer.add_scalar('train_' + k, v, i)

            self.val_loop()
            val_loss = self.val_loss_tracker.update_loss_history()
            for k,v in val_loss:
                self.writer.add_scalar('val_' + k, v, i)
            val_metrics = self.val_metrics_tracker.update_metrics()
            for k,v in val_metrics:
                if len(v) > 1:
                    self.writer.add_scalars('val_' + k, dict(zip(np.arange(len(v)), v)), i)
                else:
                    self.writer.add_scalar('val_' + k, v, i)

            if (i%checkpoint) == 0:
                filename = model_name + '_epoch_' + str(i)
                path = os.path.join(save_dir, filename)
                torch.save(self.model, path)
                print("Saved to " + path)

            for lr_scheduler in self.lr_schedulers:
                if type(lr_scheduler) == lr_scheduler.ReduceLROnPlateau:
                    lr_scheduler.step(metric=self.val_loss_tracker['total'])
                else:
                    lr_scheduler.step()
        self.writer.flush()
        self.writer.close()
        if ((num_epochs-1)%checkpoint) != 0:
            filename = model_name + '_epoch_' + str(num_epochs-1)
            path = os.path.join(save_dir, filename)
            torch.save(self.model, path)
            print("Saved to " + path)

# if __name__ == "__main__":
#     dataloaders = {"train": , "val": }

#     model = ModelWrapper(,resnet50(weights=ResNet50_Weights.DEFAULT), False)
    
#     loss_types = {'embedding': {}, 'logit': {}}
#     loss_types['embedding']['triplet'] = loss.TripletLoss(miner=miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard"), triplet_loss=losses.TripletMarginLoss())
#     loss_types['logit']['cross_entropy'] = nn.CrossEntropyLoss()
#     loss_weights = {'triplet': 1.0, 'cross_entropy': 1.0}
#     criterion = loss.ModelLoss(n, loss_types, loss_weights, False, False)

#     optimizers = [optim.Adam(model.parameters(), lr=0.01)]
#     lr_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizers[0])]
#     device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

#     train_metric_tracker = MetricTracker([metrics.MAP_K(5,use_min=True), MRR()], [metrics.MAP_K(5,use_min=True), MRR()])
#     val_metric_tracker = MetricTracker([metrics.MAP_K(5,use_min=True), MRR()], [metrics.MAP_K(5,use_min=True), MRR()])

#     trainer = Trainer(device, model, , True, optimizers, lr_schedulers, criterion)

