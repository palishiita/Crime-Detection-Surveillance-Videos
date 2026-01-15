from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import datasets, transforms
from torchvision.utils import save_image

# allow `python tools/...` run from repo root
import sys
sys.path.append(".")

from src.data.dataset import CLASSES
from src.models.model import build_mobilenetv2, build_resnet50, build_vgg16

@dataclass
class CamResult:
    cam: torch.Tensor          # (H, W), normalized [0,1]
    pred_idx: int
    pred_prob: float


class GradCAM:
    """
    Minimal Grad-CAM for CNN backbones.
    Hooks a conv layer, stores activations + gradients, produces CAM.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self._activations = None
        self._grads = None

        def fwd_hook(_, __, output):
            self._activations = output

        def bwd_hook(_, grad_in, grad_out):
            # grad_out[0] is gradient w.r.t. output of the layer
            self._grads = grad_out[0]

        self._h1 = target_layer.register_forward_hook(fwd_hook)
        self._h2 = target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> CamResult:
        """
        x: (1,3,H,W) float tensor
        class_idx: if None, uses predicted class
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # (1, C)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())

        target = pred_idx if class_idx is None else int(class_idx)
        score = logits[0, target]
        score.backward(retain_graph=False)

        if self._activations is None or self._grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/grads. Check target layer.")

        # activations/grads shape: (1, K, h, w)
        A = self._activations
        G = self._grads
        weights = G.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = (weights * A).sum(dim=1, keepdim=False)  # (1, h, w)
        cam = torch.relu(cam)

        # normalize to [0,1]
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # upsample to input size
        cam = torch.nn.functional.interpolate(
            cam[None, None, ...], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )[0, 0]

        return CamResult(cam=cam.detach().cpu(), pred_idx=pred_idx, pred_prob=pred_prob)


# -------------------------
# Model / layer selection
# -------------------------
def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    name = model_name.lower().strip()
    if name == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, dropout=0.3, freeze_backbone=False)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, dropout=dropout, freeze_backbone=False)
    if name == "vgg16":
        return build_vgg16(num_classes=num_classes, dropout=dropout, freeze_backbone=False)
    raise ValueError(f"Unknown model: {model_name}")


def pick_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Picks a reasonable "last conv" layer for Grad-CAM depending on architecture.
    """
    name = model_name.lower().strip()
    if name == "resnet50":
        # last block conv output
        return model.layer4[-1].conv3  # standard for torchvision resnet50
    if name == "mobilenetv2":
        # last conv in features
        # features is Sequential; last item is ConvBNReLU
        return model.features[-1][0] if isinstance(model.features[-1], nn.Sequential) else model.features[-1]
    if name == "vgg16":
        # last conv layer in features
        # walk backwards to find Conv2d
        for m in reversed(list(model.features.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d layer found in VGG features.")
    raise ValueError(f"Unknown model for target layer: {model_name}")


# Utilities

def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        # fallback if you saved raw state_dict
        sd = ckpt
    model.load_state_dict(sd, strict=True)


def overlay_cam_on_image(img_t: torch.Tensor, cam: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    """
    img_t: (3,H,W) in [0,1]
    cam: (H,W) in [0,1]
    Returns: (3,H,W) overlay
    """
    cam3 = cam.unsqueeze(0).repeat(3, 1, 1)
    # simple overlay: brighten where cam is high
    overlay = (1 - alpha) * img_t + alpha * torch.clamp(img_t + cam3, 0.0, 1.0)
    return overlay


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg16"])
    p.add_argument("--data_root", required=True, type=str, help="ImageFolder root: dataset/<class>/*")
    p.add_argument("--out_dir", default="results/gradcam", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--dropout", default=0.4, type=float)
    p.add_argument("--num_images", default=24, type=int, help="How many random images to visualize")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--alpha", default=0.45, type=float, help="overlay strength")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--target_class", default=None, type=str, help="Optional class name to force CAM (e.g., 'Robbery')")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = ensure_dir(args.out_dir)

    # Dataset (ImageFolder)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(root=args.data_root, transform=tfm)
    if len(ds) == 0:
        raise RuntimeError(f"No images found under {args.data_root}. Expected ImageFolder structure.")

    # Map class index to name for ImageFolder (might differ from your CLASSES)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    # Model
    model = build_model(args.model, num_classes=len(CLASSES), dropout=args.dropout).to(device)
    load_checkpoint_into_model(model, args.ckpt, device)
    model.eval()

    target_layer = pick_target_layer(model, args.model)
    cam_engine = GradCAM(model, target_layer, device)

    # Choose random samples
    n = min(args.num_images, len(ds))
    indices = np.random.choice(len(ds), size=n, replace=False).tolist()

    # Optional forced target class index (from your CLASSES list)
    forced_idx = None
    if args.target_class is not None:
        if args.target_class not in CLASSES:
            raise ValueError(f"--target_class '{args.target_class}' not in CLASSES={CLASSES}")
        forced_idx = CLASSES.index(args.target_class)

    # Save outputs
    for j, i in enumerate(indices):
        x, y = ds[i]  # x: (3,H,W) [0,1], y: ImageFolder label index
        x1 = x.unsqueeze(0).to(device)

        res = cam_engine(x1, class_idx=forced_idx)
        overlay = overlay_cam_on_image(x, res.cam, alpha=args.alpha)

        # filenames
        true_name = idx_to_class.get(y, str(y))
        pred_name = CLASSES[res.pred_idx] if res.pred_idx < len(CLASSES) else str(res.pred_idx)
        fname = f"{j:03d}_true-{true_name}_pred-{pred_name}_p{res.pred_prob:.2f}.png"

        save_image(overlay, os.path.join(out_dir, fname))

    cam_engine.close()
    print(f"Saved {n} Grad-CAM overlays to: {out_dir}")


if __name__ == "__main__":
    main()