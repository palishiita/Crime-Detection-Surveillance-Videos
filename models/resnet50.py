import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 6, dropout: float = 0.4) -> nn.Module:
    """
    Builds a ResNet50 model with a custom classification head.
    """
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Get number of features from the final FC layer
    in_features = model.fc.in_features

    # Replace final classifier
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

    return model
