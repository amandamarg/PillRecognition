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

def train_epoch(dataloader, device, models, miner, loss_funcs, loss_weights, optimizers, loss_dict, epoch_embedding_metrics, epoch_logit_metrics, clip_gradients=True):
    
    models["embedding"].train()
    models["classifier"].train()

    running_loss = {'embedding': [], 'classifier': [], 'total': []}
    embedding_metrics = {'map_1': [], 'map_5': [], 'MRR': []}
    logit_metrics = {'top_1_accuracy': [], 'top_5_accuracy': [], 'MRR': []}

    with torch.set_grad_enabled(True):
        for i,data in enumerate(tqdm(dataloader, total=len(dataloader))):
            imgs = data[0].to(device)
            labels = data[1].to(device)
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

            imgs = data[0].to(device)
            labels = data[1].to(device)

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
