from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# allow running from repo root
import sys
sys.path.append(".")

from src.data.dataset import CLASSES
from src.models.model import build_mobilenetv2, build_resnet50, build_vgg16


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    name = model_name.lower().strip()
    if name == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, dropout=0.3, freeze_backbone=False)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, dropout=dropout, freeze_backbone=False)
    if name == "vgg16":
        return build_vgg16(num_classes=num_classes, dropout=dropout, freeze_backbone=False)
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Optional[str], device: torch.device) -> None:
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)


def get_feature_extractor(model: nn.Module, model_name: str) -> nn.Module:
    """
    Returns a module that maps input -> feature vector.
    We remove the final classifier head.
    """
    name = model_name.lower().strip()
    if name == "resnet50":
        # ResNet: features are after avgpool; easiest: replace fc with Identity
        model.fc = nn.Identity()
        return model
    if name == "mobilenetv2":
        # MobileNetV2: classifier is head, features before classifier
        model.classifier = nn.Identity()
        return model
    if name == "vgg16":
        # VGG: classifier is head, features+avgpool+flatten then classifier
        model.classifier = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {model_name}")


def extract_features(
    feature_model: nn.Module,
    data_root: str,
    device: torch.device,
    img_size: int,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(root=data_root, transform=tfm)
    if len(ds) == 0:
        raise RuntimeError(f"No images found under {data_root}. Expected ImageFolder structure.")

    # sample subset
    n = min(max_samples, len(ds))
    idx = np.random.choice(len(ds), size=n, replace=False)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats_all = []
    ys_all = []

    feature_model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = feature_model(x)

            # ensure 2D feature vectors
            if f.ndim == 4:
                f = torch.mean(f, dim=(2, 3))
            elif f.ndim > 2:
                f = f.view(f.size(0), -1)

            feats_all.append(f.detach().cpu().numpy())
            ys_all.append(y.detach().cpu().numpy())

    X = np.concatenate(feats_all, axis=0).astype(np.float32)
    y = np.concatenate(ys_all, axis=0).astype(np.int64)
    return X, y


def h_score(X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """
    H-score (Bao et al.-style separability):
      H = tr( S_b * pinv(S_w) )
    where S_b is between-class scatter, S_w is within-class scatter.

    This implementation uses covariance estimates with small regularization.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    classes = np.unique(y)

    # within-class scatter
    Sw = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    # between-class scatter
    Sb = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    mu = X.mean(axis=0, keepdims=True).astype(np.float64)

    for c in classes:
        Xk = X[y == c].astype(np.float64)
        if Xk.shape[0] < 2:
            continue
        muk = Xk.mean(axis=0, keepdims=True)
        Xk0 = Xk - muk
        Sw += (Xk0.T @ Xk0)

        nk = Xk.shape[0]
        dk = (muk - mu)
        Sb += nk * (dk.T @ dk)

    # regularize Sw for numerical stability
    Sw = Sw + eps * np.eye(Sw.shape[0], dtype=np.float64)
    # compute trace(Sb * inv(Sw))
    inv_Sw = np.linalg.pinv(Sw)
    H = float(np.trace(Sb @ inv_Sw))
    return H


def linear_probe_accuracy(
    X: np.ndarray, y: np.ndarray, seed: int = 0, train_frac: float = 0.8, reg: float = 1e-4
) -> float:
    """
    Simple linear probe (ridge regression to one-hot, then argmax).
    No sklearn dependency.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # standardize
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    C = int(np.max(y) + 1)
    Y = np.eye(C, dtype=np.float32)[ytr]  # (ntr, C)

    # ridge solution: W = (X^T X + reg I)^-1 X^T Y
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Y
    W = np.linalg.solve(XtX + reg * np.eye(XtX.shape[0], dtype=np.float32), XtY)

    logits = Xte @ W
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == yte))
    return acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg16"])
    p.add_argument("--ckpt", default=None, type=str, help="Optional fine-tuned checkpoint. Omit to use ImageNet-only features.")
    p.add_argument("--data_root", required=True, type=str)
    p.add_argument("--out_dir", default="results/transferability", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--dropout", default=0.4, type=float)
    p.add_argument("--max_samples", default=4000, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    out_dir = ensure_dir(args.out_dir)

    model = build_model(args.model, num_classes=len(CLASSES), dropout=args.dropout).to(device)
    load_checkpoint_into_model(model, args.ckpt, device)

    feat_model = get_feature_extractor(model, args.model).to(device)

    X, y = extract_features(
        feature_model=feat_model,
        data_root=args.data_root,
        device=device,
        img_size=args.img_size,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    H = h_score(X, y)
    lp_acc = linear_probe_accuracy(X, y, seed=args.seed)

    # Save artifacts
    np.savez_compressed(os.path.join(out_dir, "features.npz"), X=X, y=y)
    out = {
        "model": args.model,
        "ckpt": args.ckpt,
        "num_samples": int(X.shape[0]),
        "feat_dim": int(X.shape[1]),
        "h_score": float(H),
        "linear_probe_acc": float(lp_acc),
        "note": "H-score measures class separability of features; linear probe accuracy is a practical transferability proxy."
    }
    with open(os.path.join(out_dir, "transferability.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved transferability analysis to:", out_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
