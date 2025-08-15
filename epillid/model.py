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


class MyModel(nn.Module):
    def __init__(self, n_classes, embedding_model):
        super(MyModel, self).__init__()
        self.embedding_size = embedding_model.fc.in_features
        embedding_model.fc = torch.nn.Identity()
        self.embedding_model = embedding_model
        self.classifier = Classifier(embedding_size, n_classes)
    
    def forward(self, input):
        embeddings = self.embedding_model(input)
        logits = self.classifier(input)
        return embedding, logits

