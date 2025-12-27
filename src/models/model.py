import torch.nn as nn
from torchvision import models


def _freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def build_mobilenetv2(num_classes: int = 6, dropout: float = 0.3, freeze_backbone: bool = False) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        _freeze(model)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def build_resnet50(num_classes: int = 6, dropout: float = 0.4, freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        _freeze(model)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def build_vgg16(num_classes: int = 6, dropout: float = 0.5, freeze_backbone: bool = False) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    if freeze_backbone:
        _freeze(model)

    in_features = model.classifier[6].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model