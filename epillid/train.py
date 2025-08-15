import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
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

import eval

def train_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, optimizers, loss_dict, clip_gradients=True):
    
    embedding_model.train()
    classifier.train()

    running_loss = {'embedding': 0.0, 'classifier': 0.0, 'total': 0.0}
    curr_loss = {'embedding': 0.0, 'classifier': 0.0}
    embedding_metrics = []
    logit_metrics = []

    num_batches = 0
    with torch.set_grad_enabled(True):
        for data in tqdm(dataloader, total=len(dataloader)):
            num_batches += 1

            imgs = data[0].to(device)
            labels = data[1].to(device)

            optimizers['embedding'].zero_grad()
            optimizers['classifier'].zero_grad()

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            curr_loss['embedding'] = loss_funcs['embedding'](embeddings, labels, mined_output)
            running_loss['embedding'] += curr_loss['embedding']
            embedding_metrics.append(eval.eval_embeddings(labels.cpu(), embeddings.cpu()))

            
            logits = models['classifier'](embeddings)
            curr_loss['classifier'] = loss_funcs['classifier'](logits, labels)
            running_loss['classifier'] += curr_loss['classifier']

            logit_metrics.append(eval.eval_logits(labels.cpu(), logits.cpu()))

            total_loss = get_total_loss(curr_loss, loss_weights)
            total_loss.backward()

            running_loss['total'] += total_loss

            if clip_gradients:
                nn.utils.clip_grad_norm_(models['embedding'].parameters(), 1.)
                nn.utils.clip_grad_norm_(models['classifier'].parameters(), 1.)
                
            optimizers['embedding'].step()
            optimizers['classifier'].step()

    for k, v in running_loss.items():
        loss_dict[k].append(v/num_batches)
        
    return

def val_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, loss_dict):
    embedding_model.eval()
    classifier.eval()

    running_loss = {'embedding': 0.0, 'classifier': 0.0, 'total': 0.0}
    curr_loss = {'embedding': 0.0, 'classifier': 0.0}

    embedding_metrics = []
    logit_metrics = []

    num_batches = 0
    with torch.set_grad_enabled(False):
        for data in tqdm(dataloader, total=len(dataloader)):
            num_batches += 1

            imgs = data[0].to(device)
            labels = data[1].to(device)

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            curr_loss['embedding'] = loss_funcs['embedding'](embeddings, labels, mined_output)
            running_loss['embedding'] += curr_loss['embedding']
            embedding_metrics.append(eval.eval_embeddings(labels.cpu(), embeddings.cpu()))

            logits = models['classifier'](embeddings)
            curr_loss['classifier'] = loss_funcs['classifier'](logits, labels)
            running_loss['classifier'] += curr_loss['classifier']
            logit_metrics.append(eval.eval_logits(labels.cpu(), logits.cpu()))

            running_loss['total'] += get_total_loss(curr_loss, loss_weights)
    
    for k, v in running_loss.items():
        loss_dict[k].append(v/num_batches)
        
    return

def train(models, dataloaders, num_epochs, device,  miner, loss_funcs, loss_weights, optimizers, lr_scheduler):
    models["embedding"].to(device)
    models["classifier"].to(device)

    train_loss_avgs = {"embedding": [], "classifier": [], "total": []}
    val_loss_avgs = {"embedding": [], "classifier": [], "total": []}

    for i in range(num_epochs):
        print("Training Epoch {:d}...".format(i))
        train_epoch(dataloaders["train"], device, models, miner, loss_funcs, loss_weights, optimizers, train_loss_avgs)
        print("Training loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(train_loss_avgs["embedding"][-1], train_loss_avgs["classifier"][-1], train_loss_avgs["total"][-1]))
        val_epoch(dataloaders["val"], device, models, miner, loss_funcs, loss_weights, val_loss_avgs)
        print("Validation loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(val_loss_avgs["embedding"][-1], val_loss_avgs["classifier"][-1], val_loss_avgs["total"][-1]))
        lr_scheduler["embedding"].step(val_loss_avgs["embedding"][-1])
        lr_scheduler["classifier"].step(val_loss_avgs["classifier"][-1])


