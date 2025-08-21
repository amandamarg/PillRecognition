import torch
import torch.nn as nn
from pytorch_metric_learning import losses
import numpy as np

class ModelLoss(nn.Module):
    def __init__(self, n_classes, loss_types, loss_weights, miner=None, split_embeddings=False, shift_back_labels=False):
        super(ModelLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.miner = miner
        self.split_embeddings = split_embeddings #whether we should split embeddings to calculate embedding loss
        self.shift_back_labels = shift_back_labels #whether we should shift back_labels
    
    def forward(self, embeddings, logits, labels, is_front=None):
        device = embeddings.device.type
        assert logits.device.type == device and labels.device.type == device
        if is_front is not None:
            assert is_front.device.type == device
            
        if self.split_embeddings:
            classifier_loss = self.loss_types['classifier'](logits, labels)
            embeddings = torch.vstack(embeddings.hsplit(2))
            labels = labels.clone().detach()
            labels = torch.hstack((labels, self.n_classes + labels)) if self.shift_back_labels else torch.hstack((labels, labels))
        else:
            if self.shift_back_labels:
                assert is_front is not None
                labels = labels.clone().detach()
                labels = torch.where(is_front, labels, labels+self.n_classes) 
            classifier_loss = self.loss_types['classifier'](logits, labels)

        if self.miner is not None:
            mined_output = self.miner(embeddings, labels)
            embedding_loss = self.loss_types['embedding'](embeddings, labels, mined_output)
        else:
            embedding_loss = self.loss_types['embedding'](embeddings, labels)
        total_loss = self.loss_weights['embedding']*embedding_loss.clone().detach() + self.loss_weights['classifier']*classifier_loss.clone().detach()
        return {"embedding": embedding_loss, "classifier": classifier_loss, "total": total_loss}

class LossTracker:
    def __init__(self, prev_loss_history=None):
        self.curr_epoch = 0
        self.loss_history = {"embedding": [], "classifier": [], "total": []}
        if prev_loss_history is not None:
            for k,v in prev_loss_history.items():
                self.loss_history[k].extend(v)
            self.curr_epoch = len(prev_loss_history["total"])
        self.curr_loss = {"embedding": [], "classifier": [], "total": []}
        
    def get_loss_history(self, epoch=None):
        if epoch is None:
            return self.loss_history
        else:
            output = {}
            for k,v in self.loss_history.items():
                output[k] = v[epoch]
            return output

    def best_loss(self):
        best_epoch = {}
        best_val = {}
        for k,v in self.loss_history.items():
            if len(v) > 0:
                best_epoch[k] = np.argmin(v)
                best_val[k] = np.min(v)
        return best_epoch, best_val
    
    def update_curr_loss(self, loss_type, loss_val):
        self.curr_loss[loss_type].append(loss_val.clone().detach().item())

    def update_loss_history(self):
        for k,v in self.curr_loss.items():
            self.loss_history[k].append(np.sum(v)/len(v))
            self.curr_loss[k] = []
        self.curr_epoch += 1