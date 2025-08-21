import torch
import torch.nn as nn
from pytorch_metric_learning import losses

class ModelLoss(nn.Module):
    def __init__(self, n_classes, loss_types, loss_weights, miner=None, split_embeddings=False, shift_back_labels=False):
        super(ModelLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.miner = miner
        self.split_embeddings = split_embeddings #whether we should split embeddings to calculate embedding loss
        self.shift_back_labels = shift_back_labels
    
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




        