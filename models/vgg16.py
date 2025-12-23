import torch.nn as nn
from torchvision import models


def build_vgg16(num_classes: int = 6, dropout: float = 0.5) -> nn.Module:
    """
    Builds a VGG16 model with a custom classification head.
    """

    # Load pretrained VGG16
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze feature extractor if you want (optional)
    # for param in model.features.parameters():
    #     param.requires_grad = False

    # Get input size of classifier
    in_features = model.classifier[6].in_features

    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

    return model
