import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, scaler=1.0):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_size))
        self.scaler = scaler
    
    def forward(self, input):
        normalized_input = nn.functional.normalize(input, 2, dim=1)
        return self.fc(normalized_input * self.scaler)


class ModelWrapper(nn.Module):
    def __init__(self, n_classes, embedding_model, two_sided):
        super(ModelWrapper, self).__init__()
        self.two_sided = two_sided
        self.embedding_size = embedding_model.fc.in_features
        #if two_sided, classifier will take in the concatenation of the embedding of the front-side image and the embedding of the back-side image
        if self.two_sided:
            self.embedding_size = 2*self.embedding_size
        self.n_classes = n_classes
        embedding_model.fc = torch.nn.Identity()
        self.embedding_model = embedding_model
        self.classifier = Classifier(self.embedding_size, n_classes)

    def forward(self, input):
        if self.two_sided:
            front_embeddings = self.embedding_model(input[0])
            back_embeddings = self.embedding_model(input[1])
            embeddings = torch.hstack([front_embeddings, back_embeddings])
            logits = self.classifier(embeddings)
            return embeddings, logits
        embeddings = self.embedding_model(input)
        logits = self.classifier(embeddings)
        return embeddings, logits
    
    def split_embeddings(self, embeddings, labels=None, shift=False):
        if self.two_sided:
            embeddings = torch.vstack(embeddings.hsplit(2))
            if labels is not None:
                labels = torch.hstack((labels, self.n_classes + labels)) if shift else torch.hstack((labels, labels))
        return embeddings, labels

    def shift_labels(self, labels, is_front=None):
        if is_front is None:
            return labels + self.n_classes
        return torch.where(is_front, labels, labels + self.n_classes)


    
    
    