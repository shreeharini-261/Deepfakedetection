# models/resnet50.py

import torch
import torchvision.models as models

def load_resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Output layer for binary classification
    return model