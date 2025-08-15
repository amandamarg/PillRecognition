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
import utils

def train_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, optimizers, loss_dict, epoch_embedding_metrics, epoch_logits_metrics, clip_gradients=True):
    
    embedding_model.train()
    classifier.train()

    running_loss = {'embedding': [], 'classifier': [], 'total': []}
    curr_loss = {'embedding': 0.0, 'classifier': 0.0}
    embedding_metrics = {'ap_1': [], 'ap_5': [], 'map_1': [], 'map_5': [], 'MRR': []}
    logit_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}


    with torch.set_grad_enabled(True):
        for data in tqdm(dataloader, total=len(dataloader)):
            imgs = data[0].to(device)
            labels = data[1].to(device)

            optimizers['embedding'].zero_grad()
            optimizers['classifier'].zero_grad()

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            running_loss['embedding'].append(loss_funcs['embedding'](embeddings, labels, mined_output))
            embedding_metrics.append(eval.eval_embeddings(labels.cpu(), embeddings.cpu()))
            batch_embedding_metrics = eval.eval_embeddings(labels.cpu(), embeddings.cpu())
            for k,v in batch_embedding_metrics.items():
                embedding_metrics[k].append(v)

            
            logits = models['classifier'](embeddings)
            running_loss['classifier'].append(loss_funcs['classifier'](logits, labels))
            batch_logit_metrics = eval.eval_logits(labels.cpu(), logits.cpu())
            for k,v in batch_logit_metrics.items():
                logit_metrics[k].append(v)
                
            total_loss = 0.0
            for k,v in loss_weights.items():
                total_loss += (running_loss[k][-1] * v)
            total_loss.backward()

            running_loss['total'].append(total_loss)

            if clip_gradients:
                nn.utils.clip_grad_norm_(models['embedding'].parameters(), 1.)
                nn.utils.clip_grad_norm_(models['classifier'].parameters(), 1.)
                
            optimizers['embedding'].step()
            optimizers['classifier'].step()

    for k, v in running_loss.items():
        loss_dict[k].append(v/len(v))
    
    for k, v in embedding_metrics.items():
        epoch_embedding_metrics[k].append(v/len(v))
    
    for k, v in logit_metrics.items():
        epoch_logit_metrics[k].append(v/len(v))

    return

def val_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, epoch_embedding_metrics, epoch_logits_metrics, loss_dict):
    embedding_model.eval()
    classifier.eval()

    running_loss = {'embedding': [], 'classifier': [], 'total': []}
    curr_loss = {'embedding': 0.0, 'classifier': 0.0}
    embedding_metrics = {'ap_1': [], 'ap_5': [], 'map_1': [], 'map_5': [], 'MRR': []}
    logit_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}


    with torch.set_grad_enabled(False):
        for data in tqdm(dataloader, total=len(dataloader)):

            imgs = data[0].to(device)
            labels = data[1].to(device)

            embeddings = models['embedding'](imgs)
            mined_output = miner(embeddings, labels)
            running_loss['embedding'].append(loss_funcs['embedding'](embeddings, labels, mined_output))
            batch_embedding_metrics = eval.eval_embeddings(labels.cpu(), embeddings.cpu())
            for k,v in batch_embedding_metrics.items():
                embedding_metrics[k].append(v)

            logits = models['classifier'](embeddings)
            running_loss['classifier'].append(loss_funcs['classifier'](logits, labels))
            batch_logit_metrics = eval.eval_logits(labels.cpu(), logits.cpu())
            for k,v in batch_logit_metrics.items():
                logit_metrics[k].append(v)

            total_loss = 0.0
            for k,v in loss_weights.items():
                total_loss += (running_loss[k][-1] * v)
            running_loss['total'].append(total_loss)

    for k, v in running_loss.items():
        loss_dict[k].append(v/len(v))
    
    for k, v in embedding_metrics.items():
        epoch_embedding_metrics[k].append(v/len(v))
    
    for k, v in logit_metrics.items():
        epoch_logit_metrics[k].append(v/len(v))

    return

def train(model_name, models, dataloaders, num_epochs, device,  miner, loss_funcs, loss_weights, optimizers, lr_scheduler):
    models["embedding"].to(device)
    models["classifier"].to(device)

    train_loss_avgs = {"embedding": [], "classifier": [], "total": []}
    val_loss_avgs = {"embedding": [], "classifier": [], "total": []}
    train_epoch_embedding_metrics = {'ap_1': [], 'ap_5': [], 'map_1': [], 'map_5': [], 'MRR': []}
    train_epoch_logits_metrics = {}
    val_epoch_embedding_metrics = {'ap_1': [], 'ap_5': [], 'map_1': [], 'map_5': [], 'MRR': []}
    val_epoch_logits_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}

    for i in range(num_epochs):
        print("Training Epoch {:d}...".format(i))
        train_epoch(dataloaders["train"], device, models, miner, loss_funcs, loss_weights, optimizers, train_loss_avgs, train_epoch_embedding_metrics, train_epoch_logits_metrics, True)
        print("Training loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(train_loss_avgs["embedding"][-1], train_loss_avgs["classifier"][-1], train_loss_avgs["total"][-1]))
        val_epoch(dataloaders["val"], device, models, miner, loss_funcs, loss_weights, val_loss_avgs, val_epoch_embedding_metrics, val_epoch_logits_metrics)
        print("Validation loss: embedding_model_loss={:f}, classifier_loss={:f}, total={:f}".format(val_loss_avgs["embedding"][-1], val_loss_avgs["classifier"][-1], val_loss_avgs["total"][-1]))
        lr_scheduler["embedding"].step(val_loss_avgs["embedding"][-1])
        lr_scheduler["classifier"].step(val_loss_avgs["classifier"][-1])
    
    utils.save_model(model_name, num_epochs)


