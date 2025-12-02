from torchvision import transforms

def get_transforms():
    """
    Returns preprocessing transforms for all images.
    Matches the methodology: resizing + normalization only.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),   # resize frames to model input size
        transforms.ToTensor(),            # convert PIL image to tensor
        transforms.Normalize(             # ImageNet normalization for pretrained models
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])