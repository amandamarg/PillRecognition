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

from model import Classifier
from dataset import PillImages, CustomBatchSamplerPillID
from train import train
import utils


LABELCOL = 'label'
TRAIN_SIDE_LABELS = True
NUM_EPOCHS = 10
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
    

    datasets = {k: PillImages(pd.concat([v , ref_df]).reset_index(drop=True), k, labelcol=encoded_label_name) for k,v in cons_splits}
    train_batch_sampler = CustomBatchSamplerPillID(datasets['train'].df, batch_size=BATCH_SIZE, labelcol='encoded_label')
    val_batch_sampler = CustomBatchSamplerPillID(datasets['val'].df, batch_size=BATCH_SIZE, labelcol='encoded_label')
    train_dataloader = DataLoader(datasets['train'], batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(datasets['val'], batch_sampler=val_batch_sampler)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

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
    print(train_metrics)
    print(val_metrics)


