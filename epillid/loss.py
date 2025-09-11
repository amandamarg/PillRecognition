import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
import numpy as np

class ModelLoss(nn.Module):
    def __init__(self, n_classes, loss_types, loss_weights, split_embeddings=False, shift_side_labels=False):
        super(ModelLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.split_embeddings = split_embeddings #whether we should split embeddings to calculate embedding loss
        self.shift_side_labels = shift_side_labels #whether we should shift labels of backside images
    
    def forward(self, embeddings, logits, labels, is_front=None, is_ref=None):
        device = embeddings.device.type
        assert logits.device.type == device and labels.device.type == device
        if is_front is not None:
            assert is_front.device.type == device
        
        losses = {}
        weighted_loss = torch.Tensor([0.0]).to(device)
        if self.split_embeddings:
            for loss_name,loss_func in self.loss_types['logit'].items():
                losses[loss_name] = loss_func(logits, labels)
                weighted_loss += (self.loss_weights[loss_name] * losses[loss_name])
            embeddings = torch.vstack(embeddings.hsplit(2))
            labels = labels.clone().detach()
            labels = torch.hstack((labels, self.n_classes + labels)) if self.shift_side_labels else torch.hstack((labels, labels))
        else:
            if self.shift_side_labels:
                assert is_front is not None
                labels = labels.clone().detach()
                labels = torch.where(is_front, labels, labels+self.n_classes) 
            for loss_name,loss_func in self.loss_types['logit'].items():
                losses[loss_name] = loss_func(logits, labels)
                weighted_loss += (self.loss_weights[loss_name] * losses[loss_name])

        for loss_name,loss_func in self.loss_types['embedding'].items():
            losses[loss_name] = loss_func(embeddings, labels)
            weighted_loss += (self.loss_weights[loss_name] * losses[loss_name])
        
        losses['total'] = weighted_loss
        return losses
# class TripletLoss(nn.Module):
#     def __init__(self, miner, triplet_loss):
#         super(TripletLoss, self).__init__()
#         self.miner = miner
#         self.triplet_loss = triplet_loss

#     def forward(self, embeddings, labels):
#         triplets = self.miner(embeddings, labels)
#         return self.triplet_loss(embeddings, labels, triplets)
from triplets import get_hardest_triplets, get_triplets, triplet_selector
import torch.nn.functional as F
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, mode='all', use_cosine_dist=False):
        super(TripletLoss, self).__init__()
        self.margin=margin
        self.mode=mode
        self.use_cosine_dist=use_cosine_dist

    def get_triplets(self, embeddings, labels, is_front=None, is_ref=None):
        # if self.mode == 'hardest':
        #     triplets, pos_dist, neg_dist = get_hardest_triplets(embeddings, labels, is_front, is_ref, self.use_cosine_dist)
        #     return triplets, pos_dist, neg_dist
        triplets = get_triplets(labels, is_front, is_ref)
        triplets, pos_dist, neg_dist = triplet_selector(embeddings, triplets, self.margin, self.mode, self.use_cosine_dist)
        return triplets, pos_dist, neg_dist

    def forward(self, inputs):
        embeddings = inputs['embeddings']
        labels = inputs['labels']
        is_front = inputs.get('is_front', None)
        is_ref = inputs.get('is_ref', None)
        triplets, pos_dists, neg_dists = self.get_triplets(embeddings, labels, is_front, is_ref)
        return F.relu(pos_dists-neg_dists+self.margin).mean()


class LossWrapper(nn.Module):
    def __init__(self, loss_types, loss_weights, device):
        super(LossWrapper, self).__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.device = device

    def forward(self, inputs):
        losses = {}
        weighted_loss = torch.Tensor([0.0]).to(self.device)
        
        for loss_name,loss_func in self.loss_types.items():
            losses[loss_name] = loss_func(inputs)
            weighted_loss += (self.loss_weights[loss_name] * losses[loss_name])
        
        losses['total'] = weighted_loss
        return losses
# class LossTracker:
#     def __init__(self):
#         self.loss_history = {}
#         self.curr_loss = {}

#     def best_loss(self):
#         best_epoch = {}
#         best_val = {}
#         for k,v in self.loss_history.items():
#             if len(v) > 0:
#                 best_epoch[k] = np.argmin(v)
#                 best_val[k] = np.min(v)
#             else:
#                 best_epoch[k] = None
#                 best_val[k] = None
#         return best_epoch, best_val
    
#     def update_curr_loss(self, loss_type, loss_val):
#         if loss_type in self.loss_history.keys():
#             self.curr_loss[loss_type].append(loss_val.item())
#         else:
#             self.curr_loss[loss_type] = [loss_val.item()]

#     def update_loss_history(self):
#         outputs = {}
#         for k,v in self.curr_loss.items():
#             outputs[k] = np.sum(v)/len(v)
#             if k not in self.loss_history.keys():
#                 self.loss_history[k] = [outputs[k]]
#             else:
#                 self.loss_history[k].append(outputs[k])
#             self.curr_loss[k] = []
#         return outputs

class LossTracker:
    def __init__(self):
        self.loss_history = {}
        self.curr_loss = {}

    def best_loss(self):
        best_epoch = {}
        best_val = {}
        for k,v in self.loss_history.items():
            if len(v) > 0:
                best_epoch[k] = torch.argmin(v)
                best_val[k] = torch.min(v)
            else:
                best_epoch[k] = None
                best_val[k] = None
        return best_epoch, best_val
    
    def update_curr_loss(self, loss):
        for k,v in loss.items():
            if k in self.curr_loss.keys():
                self.curr_loss[k] = torch.concat((self.curr_loss[k], v.detach().cpu().ravel()))
            else:
                self.curr_loss[k] = v.detach().cpu().ravel()

    def update_loss_history(self):
        outputs = {}
        for k,v in self.curr_loss.items():
            outputs[k] = torch.mean(v)
            if k not in self.loss_history.keys():
                self.loss_history[k] = outputs[k].ravel()
            else:
                self.loss_history[k] = torch.concat((self.loss_history[k], outputs[k].ravel()))
            self.curr_loss[k] = torch.tensor([])
        return outputs

if __name__ == "__main__":
    loss_types = {'embedding': {}, 'logit': {}}
    loss_types['embedding']['triplet'] = TripletLoss(miner=miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard"), triplet_loss=losses.TripletMarginLoss())
    loss_types['logit']['cross_entropy'] = nn.CrossEntropyLoss()
    loss_weights = {'triplet': 1.0, 'cross_entropy': 1.0}

    num_samples = 25
    num_classes = 10
    size_embedding = 100
    torch.manual_seed(0)

    sample_labels = torch.randint(0,num_classes, (num_samples,))
    sample_embeddings = torch.rand((num_samples,size_embedding))
    sample_logits = (torch.rand((num_samples,n_classes)) - .5)

    criterion = ModelLoss(n, loss_types, loss_weights, False, False)
