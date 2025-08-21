import torch
import torch.nn as nn
from pytorch_metric_learning import losses

class ModelLoss(nn.Module):
    def __init__(self, n_classes, two_sided, shift_backside_labels, loss_types, loss_weights, miner):
        super(ModelLoss, self).__init__()
        self.n_classes = n_classes
        self.two_sided = two_sided 
        self.shift_backside_labels = shift_backside_labels
        self.loss_types = loss_types
        self.loss_weights = loss_weights
    
    def forward(self, embeddings, logits, labels, is_front=None):
        if self.two_sided:
            embeddings = torch.vstack(embeddings.hsplit(2)) # split concatenated front/back embeddings
            back_labels = (labels + self.n_classes) if self.shift_backside_labels else labels
            labels = torch.hstack((labels, back_labels))
        elif self.shift_backside_labels:
            assert is_front is not None
            labels = torch.where(is_front, labels, labels + self.n+classes)
        embedding_loss = self.loss_types['embedding'](embeddings, labels)
        classifier_loss = self.loss_types['classifier'](logits, labels)
        total_loss = self.loss_weights['embedding']*embedding_loss + self.loss_weights['classifier']*classifier_loss
        return {"embedding": embedding_loss, "classifier": classifier_loss, "total": total_loss}




        