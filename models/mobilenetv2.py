import torch.nn as nn
from torchvision import models


def build_mobilenetv2(num_classes: int = 6, dropout: float = 0.3) -> nn.Module:
    """
    Builds a MobileNetV2 model with a custom classification head.
    """

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Get the number of input features of classifier
    in_features = model.classifier[1].in_features

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

    return model
