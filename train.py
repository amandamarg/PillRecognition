
import torch
from tqdm import tqdm
import torch.optim as optim
import os
from sklearn.metrics import top_k_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from benchmark.metrics import MetricsCollection
import torch.nn as nn
from evaluate import LogitsEvaluator, EmbeddingEvaluator
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dataset import CustomBatchSamplerPillID, PillImages
import utils
from benchmark.models.multihead_model import MultiheadModel
from benchmark.models.embedding_model import EmbeddingModel
from benchmark.models.losses import MultiheadLoss
from benchmark.metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector
from metrics import topk_acc

class Trainer:
    def __init__(self, device, model, dataloaders, clip_gradients, optimizer, lr_scheduler, criterion, writer, eval_update_type="logit", metric_type="euclidean", simulate_pairs=False, shift_labels=False, train_with_ref_labels=False, plot_metrics_names=["acc_1", "acc_5", "loss", "micro_ap", "map", "mrr"], path="./"):
        self.device = device
        self.model = model.to(device)
        self.n_classes = self.model.get_original_n_classes()
        self.dataloaders = dataloaders
        self.clip_gradients = clip_gradients
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.batch_metrics = MetricsCollection()
        self.epoch_metrics = MetricsCollection()
        self.writer = writer
        self.shift_labels = shift_labels & self.model.train_with_side_labels
        self.eval_update_type = eval_update_type
        self.simulate_pairs = simulate_pairs
        self.logits_evaluator = LogitsEvaluator(self.n_classes, side_aggr_mode="max", pair_aggr_mode="max", simulate_pairs=simulate_pairs, shift_labels=self.shift_labels)
        self.embedding_evaluator = EmbeddingEvaluator(self.n_classes, ref_aggr_mode="max", pair_aggr_mode="max", score_type=metric_type, simulate_pairs=simulate_pairs, shift_labels=self.shift_labels)
        self.train_with_ref_labels = train_with_ref_labels
        self.plot_metrics_names = plot_metrics_names
        self.path = path
        self.best_model_path = None
        os.makedirs(os.path.join(self.path, "checkpoints"), exist_ok=True)

    def update_batch_metrics(self, phase, labels, loss_outputs, model_outputs):
        batch_size = len(labels)
        for k in ['loss', 'metric_loss', 'ce', 'arcface', 'contrastive', 'triplet', 'focal']:
            if k in loss_outputs:
                self.batch_metrics.add(phase, k, loss_outputs[k].item(), batch_size)
        acc = topk_acc(labels, model_outputs["logits"], k_vals=[1,5])
        self.batch_metrics.add(phase, "acc_1", acc[0].item(), batch_size)
        self.batch_metrics.add(phase, "acc_5", acc[1].item(), batch_size)

    def update_epoch_metrics(self, i=1, update_plots=True):
        if update_plots:
            plot_vals = {}
        for phase in ['train', 'val']:
            for k,v in self.batch_metrics[phase].items():
                batch_avg = v.avg
                self.epoch_metrics.add(phase, k, batch_avg, 1)
                self.batch_metrics[phase][k].reset()
                if update_plots and k in self.plot_metrics_names:
                    if k not in plot_vals.keys():
                        plot_vals[k] = {}
                    plot_vals[k][phase] = batch_avg
        if update_plots:
            for k,v in plot_vals.items():
                self.writer.add_scalars(k, v, i)
    
    def train_loop(self):
        self.model.train()
        for data in tqdm(self.dataloaders["train"], total=len(self.dataloaders["train"])):
            labels = data["label"].to(self.device).long()
            imgs = data["img"].to(self.device)
            is_front = data["is_front"].to(self.device).bool() if self.model.train_with_side_labels else None
            is_ref = data["is_ref"].to(self.device).bool() if self.train_with_ref_labels else None
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model_outputs = self.model(imgs, labels)
                if self.shift_labels:
                    labels = torch.where(is_front, labels, labels + self.n_classes)
                elif self.model.train_with_side_labels:
                    model_outputs["logits"] = self.model.shift_label_indexes(model_outputs["logits"])
                loss = self.criterion(model_outputs, labels, is_front, is_ref)
                loss["loss"].backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
            self.update_batch_metrics('train', labels, loss, model_outputs)

    def val_loop(self):
        self.model.eval()
        for data in tqdm(self.dataloaders["val"], total=len(self.dataloaders["val"])):
            labels = data["label"].to(self.device).long()
            imgs = data["img"].to(self.device)
            is_front = data["is_front"].to(self.device).bool() if self.model.train_with_side_labels else None
            is_ref = data["is_ref"].to(self.device).bool() if self.train_with_ref_labels else None
            with torch.set_grad_enabled(False):
                model_outputs = self.model(imgs, labels)
                if self.shift_labels:
                    labels = torch.where(is_front, labels, labels + self.n_classes)
                elif self.model.train_with_side_labels:
                    model_outputs["logits"] = self.model.shift_label_indexes(model_outputs["logits"])
                loss = self.criterion(model_outputs, labels, is_front, is_ref)
            self.update_batch_metrics('val', labels, loss, model_outputs)

    def eval(self, i=1, plot=True):
        self.model.eval()
        outputs = {"labels": [], "is_front": [], "is_ref": []}
        print("Loading eval data...")
        for data in tqdm(self.dataloaders["eval"], total=len(self.dataloaders["eval"])):
            imgs = data["img"].to(self.device)
            labels = data["label"].long().to(self.device)
            outputs["labels"].append(labels)
            outputs["is_front"].append(data["is_front"].bool().to(self.device))
            outputs["is_ref"].append(data["is_ref"].bool().to(self.device))
            with torch.no_grad():
                model_outputs = self.model(imgs, labels)
                for k,v in model_outputs.items():
                    if k not in outputs:
                        outputs[k] = [v]
                    else:
                        outputs[k].append(v)
        for k,v in outputs.items():
            outputs[k] = torch.cat(v, 0)

        print("Starting logit eval...")
        if self.train_with_ref_labels:
            logit_eval = self.logits_evaluator.eval(outputs["logits"], outputs["labels"], outputs["is_front"], outputs["is_ref"])
        else:
            logit_eval = self.logits_evaluator.eval(outputs["logits"], outputs["labels"], outputs["is_front"], None)
        print("Starting emb eval...")
        emb_eval = self.embedding_evaluator.eval(outputs["emb"], outputs["labels"], outputs["is_front"], outputs["is_ref"])
        for k,v in logit_eval.items():
            self.epoch_metrics.add("logit_eval", k, v, 1)
            if plot and k in self.plot_metrics_names:
                self.writer.add_scalar(f"logit_{k}", v, i)
            print(f"logit_{k}={v}")
        for k,v in emb_eval.items():
            self.epoch_metrics.add("emb_eval", k, v, 1)
            if plot and k in self.plot_metrics_names:
                self.writer.add_scalar(f"emb_{k}", v, i)
            print(f"emb_{k}={v}")

    def save_checkpoint(self, i, earlystop_patience=None, has_waited=0, stop_training=False):
        checkpoint_path = os.path.join(self.path, "checkpoints", f'epoch_{i}.pth')
        print("Saving model to path " + checkpoint_path)
        torch.save(self.model, checkpoint_path)
        self.writer.flush()
        best_value, best_checkpoint_index = self.epoch_metrics[self.eval_update_type + '_eval']['micro_ap'].best(mode='max')
        #early stopping implementation from https://github.com/usuyama/ePillID-benchmark
        if best_checkpoint_index + 1 == len(self.epoch_metrics[self.eval_update_type + '_eval']['micro_ap'].history):
            has_waited = 1 if earlystop_patience is not None else 0
            print(f"Best checkpoint: {best_checkpoint_index}, Best value: {best_value}")
            self.best_model_path = checkpoint_path
        elif earlystop_patience is not None:
            if has_waited >= earlystop_patience:
                print("** Early stop in training: {} waits **".format(has_waited))
                stop_training = True
            has_waited += 1
        return best_checkpoint_index, has_waited, stop_training

    def train(self, num_epochs, checkpoint=3, earlystop_patience=None):
        has_waited = 0
        stop_training = False
        if len(self.epoch_metrics.metrics) == 0:
            num_epochs_trained = 0
        else:
            num_epochs_trained = len(self.epoch_metrics["train"]["loss"].history)
        for i in range(num_epochs_trained, num_epochs_trained + num_epochs):
            print(f"Epoch {i}")
            print("Running train loop...")
            self.train_loop()
            print("Running val loop...")
            self.val_loop()
            self.update_epoch_metrics(i=i)
            if (i%checkpoint) == 0:
                self.eval(i)
                if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.lr_scheduler.step(-1*self.epoch_metrics[self.eval_update_type + "_eval"]["micro_ap"].value)
                else:
                    self.lr_scheduler.step()
                best_checkpoint_index, has_waited, stop_training = self.save_checkpoint(i, earlystop_patience, has_waited, stop_training)
                print(f"Best Checkpoint index: {best_checkpoint_index}")
            if stop_training:
                break
        print(f"Trainning stopped at epoch {i}")
        if (i%checkpoint != 0) and not stop_training:
            self.eval(i)
            self.save_checkpoint(i)
        self.writer.close()
        return self.best_model_path