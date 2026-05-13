from pathlib import Path
from class_names import class_names
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
import torch.nn as nn

def get_model():
    model_0 = efficientnet_b0(weights=EfficientNet_B0_Weights)
    for param in model_0.features.parameters():
        param.requires_grad = False
    weights = EfficientNet_B0_Weights.DEFAULT
    model_0.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    model_0.load_state_dict(torch.load("best_model_0.pth"))
    return model_0