from models.resnet50 import build_resnet50
from models.vgg16 import build_vgg16
from models.mobilenetv2 import build_mobilenetv2


def build_model(name: str, num_classes: int = 6):
    name = name.lower()

    if name == "resnet50":
        return build_resnet50(num_classes)
    elif name == "vgg16":
        return build_vgg16(num_classes)
    elif name == "mobilenetv2":
        return build_mobilenetv2(num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
