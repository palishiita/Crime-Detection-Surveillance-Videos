from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data.dataloader import DataConfig, build_dataloaders
from src.data.dataset import CLASSES
from src.evaluation.metrics import compute_metrics
from src.evaluation.temporal import aggregate_video_predictions
from src.models.model import build_mobilenetv2, build_resnet50, build_vgg16


def build_model(model_name: str, num_classes: int, dropout: float = 0.4, freeze_backbone: bool = False) -> nn.Module:
    name = model_name.lower().strip()
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, dropout=dropout, freeze_backbone=freeze_backbone)
    if name in ("mobilenetv2", "mobilenet_v2", "mobilenet"):
        return build_mobilenetv2(num_classes=num_classes, dropout=0.3, freeze_backbone=freeze_backbone)
    if name == "vgg16":
        return build_vgg16(num_classes=num_classes, dropout=0.5, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model_name: {model_name}. Use: resnet50 | mobilenetv2 | vgg16")


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {ckpt_path}")
    return ckpt


def save_metrics_json(metrics_dict: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)


def save_classification_report_csv(per_class: Dict[str, Dict[str, float]], out_path: str) -> None:
    fieldnames = ["class", "precision", "recall", "f1", "support"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls_name, stats in per_class.items():
            writer.writerow(
                {
                    "class": cls_name,
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1"],
                    "support": stats["support"],
                }
            )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


@torch.no_grad()
def collect_logits_and_meta(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Collect per-frame logits + y_true + (video_id, frame_id) from a loader that returns meta.
    Assumes dataset returns (img, label, meta) where meta.video_id and meta.frame_id exist.
    """
    model.eval()

    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    vid_all: List[str] = []
    fid_all: List[int] = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].cpu().numpy()

        meta = batch[2]
        # default collate turns dataclass into list of SampleMeta
        for m in meta:
            vid_all.append(str(m.video_id))
            fid_all.append(int(m.frame_id))

        logits = model(x).detach().cpu().numpy()

        logits_all.append(logits)
        y_all.append(y)

    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(y_all, axis=0),
        vid_all,
        fid_all,
    )


def evaluate(
    ckpt_path: str,
    model_name: str,
    root_dir: str = "dataset",
    out_dir: str = "results",
    device_str: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    dropout: float = 0.4,
    agg_method: str = "mean_probs",
    smoothing: str = "none",
    smoothing_alpha: float = 0.7,
) -> Dict:
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # Build loaders (we only need test_loader). Ensure return_meta=True for temporal aggregation.
    data_cfg = DataConfig(
        root_dir=root_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampling=False,
        return_meta=True,
        max_per_class_train=None,
        max_per_class_test=None,
    )
    _, _, test_loader, _ = build_dataloaders(data_cfg)

    # Load model + checkpoint
    model = build_model(model_name=model_name, num_classes=len(CLASSES), dropout=dropout, freeze_backbone=False).to(device)
    ckpt = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Collect logits + meta
    logits, y_true, video_ids, frame_ids = collect_logits_and_meta(model, test_loader, device)

    # Frame-level metrics
    y_pred_frame = np.argmax(logits, axis=1)
    frame_metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred_frame,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize="true",
    )

    # Video-level metrics (temporal aggregation)
    video_res = aggregate_video_predictions(
        logits=logits,
        y_true=y_true,
        video_ids=video_ids,
        frame_ids=frame_ids,
        method=agg_method,
        smoothing=smoothing,
        smoothing_alpha=smoothing_alpha,
    )
    video_metrics = compute_metrics(
        y_true=video_res.y_true_video,
        y_pred=video_res.y_pred_video,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize="true",
    )

    # Output directory
    out_dir = Path(out_dir) / model_name
    os.makedirs(out_dir, exist_ok=True)

    # Save JSON summary (both frame + video metrics)
    metrics_json = {
        "checkpoint": ckpt_path,
        "model_name": model_name,
        "epoch": int(ckpt.get("epoch", -1)),
        "tag": ckpt.get("tag", ""),
        "best_score_in_ckpt": float(ckpt.get("best_score", float("nan"))),
        "temporal_aggregation": {
            "method": agg_method,
            "smoothing": smoothing,
            "smoothing_alpha": smoothing_alpha,
        },
        "frame_level": {
            "accuracy": frame_metrics.accuracy,
            "balanced_accuracy": frame_metrics.balanced_accuracy,
            "macro_precision": frame_metrics.macro_precision,
            "macro_recall": frame_metrics.macro_recall,
            "macro_f1": frame_metrics.macro_f1,
            "weighted_precision": frame_metrics.weighted_precision,
            "weighted_recall": frame_metrics.weighted_recall,
            "weighted_f1": frame_metrics.weighted_f1,
            "per_class": frame_metrics.per_class,
        },
        "video_level": {
            "accuracy": video_metrics.accuracy,
            "balanced_accuracy": video_metrics.balanced_accuracy,
            "macro_precision": video_metrics.macro_precision,
            "macro_recall": video_metrics.macro_recall,
            "macro_f1": video_metrics.macro_f1,
            "weighted_precision": video_metrics.weighted_precision,
            "weighted_recall": video_metrics.weighted_recall,
            "weighted_f1": video_metrics.weighted_f1,
            "per_class": video_metrics.per_class,
        },
    }
    save_metrics_json(metrics_json, str(out_dir / "metrics.json"))

    # Save per-class CSVs
    save_classification_report_csv(frame_metrics.per_class, str(out_dir / "classification_report_frame.csv"))
    save_classification_report_csv(video_metrics.per_class, str(out_dir / "classification_report_video.csv"))

    # Save confusion matrices (normalized) + raw matrices
    np.save(str(out_dir / "confusion_matrix_frame.npy"), frame_metrics.confusion_matrix)
    np.save(str(out_dir / "confusion_matrix_video.npy"), video_metrics.confusion_matrix)

    plot_confusion_matrix(
        cm=frame_metrics.confusion_matrix,
        class_names=CLASSES,
        out_path=str(out_dir / "confusion_matrix_frame.png"),
        title="Frame-level Confusion Matrix (Normalized by True Label)",
    )
    plot_confusion_matrix(
        cm=video_metrics.confusion_matrix,
        class_names=CLASSES,
        out_path=str(out_dir / "confusion_matrix_video.png"),
        title="Video-level Confusion Matrix (Temporal Aggregation, Normalized by True Label)",
    )

    return {
        "out_dir": str(out_dir),
        "metrics_json": str(out_dir / "metrics.json"),
        "classification_report_frame_csv": str(out_dir / "classification_report_frame.csv"),
        "classification_report_video_csv": str(out_dir / "classification_report_video.csv"),
        "confusion_matrix_frame_png": str(out_dir / "confusion_matrix_frame.png"),
        "confusion_matrix_video_png": str(out_dir / "confusion_matrix_video.png"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on test set with optional temporal aggregation.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenetv2", "vgg16"])
    p.add_argument("--root_dir", type=str, default="dataset")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--dropout", type=float, default=0.4)

    p.add_argument("--agg_method", type=str, default="mean_probs",
                   help="majority_vote | mean_probs | max_probs | topk_mean_probs or topk_mean_probs:k")
    p.add_argument("--smoothing", type=str, default="none", choices=["none", "ema_probs"])
    p.add_argument("--smoothing_alpha", type=float, default=0.7)
    return p.parse_args()