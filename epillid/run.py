from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.optim as optim
import torch.nn as nn
import torch 
from sklearn.preprocessing import LabelEncoder
import os
from pytorch_metric_learning import losses, miners
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import Classifier
from dataset import PillImages, CustomBatchSamplerPillID
from train import train
import utils


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


LABELCOL = 'label'
TRAIN_SIDE_LABELS = True
NUM_EPOCHS = 3
BATCH_SIZE = 32
MODEL_NAME = 'my_model'

if __name__ == "__main__":
    data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata'
    all_imgs_df, fold_indicies = utils.load_data(data_root_dir)
    all_imgs_df['image_path'] = all_imgs_df['image_path'].apply(lambda x: os.path.join(data_root_dir, 'classification_data', x))

    if TRAIN_SIDE_LABELS:
        all_imgs_df[LABELCOL] = all_imgs_df.apply(lambda x: x[LABELCOL] + '_' + ('0' if x.is_front else '1'), axis=1)
    
    unique_classes = all_imgs_df[LABELCOL].unique()
    n_classes = all_imgs_df[LABELCOL].nunique()

    label_encoder = LabelEncoder()
    encoded_label_name = 'encoded_' + LABELCOL
    all_imgs_df[encoded_label_name] = label_encoder.fit_transform(all_imgs_df[LABELCOL])
    ref_df = all_imgs_df[all_imgs_df.is_ref].reset_index(drop=True)
    cons_splits = utils.split_data(all_imgs_df, fold_indicies, val_fold=3, test_fold=4)


    datasets = {k: PillImages(pd.concat([v , ref_df]).reset_index(drop=True), k, labelcol=encoded_label_name) for k,v in cons_splits.items()}
    train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=BATCH_SIZE, labelcol='encoded_label')
    val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=BATCH_SIZE, labelcol='encoded_label')
    train_dataloader = DataLoader(datasets['train'], batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(datasets['val'], batch_sampler=val_batch_sampler)
    dataloaders = {"train": DataLoader(datasets['train'], batch_sampler=train_batch_sampler), "val": DataLoader(datasets['val'], batch_sampler=val_batch_sampler)}

    #initalize embedding model
    embedding_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    embedding_size = embedding_model.fc.in_features
    #peel off last layer
    embedding_model.fc = torch.nn.Identity()

    classifier = Classifier(embedding_size, n_classes)

    models = {"embedding": embedding_model, "classifier": classifier}

    loss_funcs = {"embedding": losses.TripletMarginLoss(), "classifier": nn.CrossEntropyLoss()}

    loss_weights = {"embedding": 1.0, "classifier": 1.0}

    optimizers = {"embedding": optim.Adam(embedding_model.parameters(), lr=0.01), "classifier": optim.Adam(classifier.parameters(), lr=0.01)}

    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

    lr_schedulers = {"embedding": optim.lr_scheduler.ReduceLROnPlateau(optimizers["embedding"], factor=0.5, patience=2), "classifier": optim.lr_scheduler.ReduceLROnPlateau(optimizers["classifier"], factor=0.5, patience=2)}

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("Using device ", device)

    train_metrics, val_metrics = train(MODEL_NAME, models, dataloaders, NUM_EPOCHS, device,  miner, loss_funcs, loss_weights, optimizers, lr_schedulers)
    



