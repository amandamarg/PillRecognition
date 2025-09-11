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

class EmbeddingModel(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingModel, self).__init__()
        self.embedding_size = base_model.fc.in_features
        base_model.fc = torch.nn.Identity()
        self.base_model = base_model

    def forward(self, input):
        return self.base_model(input)

# class ModelWrapper(nn.Module):
#     def __init__(self, n_classes, embedding_model, two_sided):
#         super(ModelWrapper, self).__init__()
#         self.two_sided = two_sided
#         self.embedding_size = embedding_model.fc.in_features
#         #if two_sided, classifier will take in the concatenation of the embedding of the front-side image and the embedding of the back-side image
#         if self.two_sided:
#             self.embedding_size = 2*self.embedding_size
#         self.n_classes = n_classes
#         embedding_model.fc = torch.nn.Identity()
#         self.embedding_model = embedding_model
#         self.classifier = Classifier(self.embedding_size, n_classes)

#     def forward(self, input):
#         if self.two_sided:
#             front_embeddings = self.embedding_model(input[0])
#             back_embeddings = self.embedding_model(input[1])
#             embeddings = torch.hstack([front_embeddings, back_embeddings])
#             logits = self.classifier(embeddings)
#             return embeddings, logits
#         embeddings = self.embedding_model(input)
#         logits = self.classifier(embeddings)
#         return embeddings, logits
    
    
import torch.nn.functional as F
class ModelWrapper(nn.Module):
    def __init__(self, n_classes, embedding_model):
        super(ModelWrapper, self).__init__()
        self.n_classes = n_classes
        self.embedding_model = embedding_model
        self.classifier = Classifier(embedding_model.embedding_size, n_classes)

    def forward(self, input, softmax=False):
        embeddings = self.embedding_model(input)
        logits = self.classifier(embeddings)
        if softmax:
            logits = F.softmax(logits, dim=1)
        return embeddings, logits