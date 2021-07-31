import torch.nn as nn
from torchvision.models import densenet169
import numpy as np

def get_classify_model():
    model = densenet169(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    in_feature = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(in_feature, 1, bias=True), nn.Sigmoid())

    return model