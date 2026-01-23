from __future__ import annotations
import json
import os
import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader
from src.data.dataset import CLASSES
from src.evaluation.metrics import compute_metrics
from src.evaluation.temporal import aggregate_video_predictions


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    cm_normalize: Optional[str] = None,
    agg_method: str = "topk_mean_probs:0.05",
    smoothing: str = "none",
    smoothing_alpha: float = 0.7,
    normal_class_name: str = "Normal",
    topk_score: str = "max",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    logits_all = []
    y_true_all = []
    video_ids = []
    frame_ids = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        metas = batch[2]

        for m in metas:
            video_ids.append(str(m.video_id))
            try:
                frame_ids.append(int(m.frame_id))
            except Exception:
                frame_ids.append(len(frame_ids))

        logits = model(x)
        logits_all.append(logits.detach().cpu().numpy())
        y_true_all.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    y_true_np = np.concatenate(y_true_all, axis=0)

    # Frame-level metrics (argmax logits == argmax softmax(logits))
    y_pred_frame = np.argmax(logits_np, axis=1)
    frame_metrics = compute_metrics(
        y_true=y_true_np,
        y_pred=y_pred_frame,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize=cm_normalize,
    )

    # Video-level aggregation + metrics
    normal_idx = None
    if topk_score == "crime_max":
        if normal_class_name in CLASSES:
            normal_idx = CLASSES.index(normal_class_name)
        else:
            raise ValueError(f"normal_class_name='{normal_class_name}' not in CLASSES={CLASSES}")

    video_res = aggregate_video_predictions(
        logits=logits_np,
        y_true=y_true_np,
        video_ids=video_ids,
        frame_ids=frame_ids,
        method=agg_method,
        smoothing=smoothing,
        smoothing_alpha=smoothing_alpha,
        normal_class_idx=normal_idx,
        topk_score=topk_score,
    )

    video_metrics = compute_metrics(
        y_true=video_res.y_true_video,
        y_pred=video_res.y_pred_video,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize=cm_normalize,
    )

    cm = video_metrics.confusion_matrix.astype(np.float64)
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    fp = np.sum(cm, axis=0) - tp
    recall_per_class = tp / np.maximum(tp + fn, 1.0)
    precision_per_class = tp / np.maximum(tp + fp, 1.0)
    video_macro_recall = float(np.mean(recall_per_class))
    video_macro_precision = float(np.mean(precision_per_class))

    results = {
        "frame": {
            "accuracy": frame_metrics.accuracy,
            "balanced_accuracy": frame_metrics.balanced_accuracy,
            "macro_precision": frame_metrics.macro_precision,
            "macro_recall": frame_metrics.macro_recall,
            "macro_f1": frame_metrics.macro_f1,
            "weighted_f1": frame_metrics.weighted_f1,
            "per_class": frame_metrics.per_class,
            "confusion_matrix": frame_metrics.confusion_matrix.tolist(),
        },
        "video": {
            "agg_method": agg_method,
            "smoothing": smoothing,
            "smoothing_alpha": smoothing_alpha,
            "accuracy": video_metrics.accuracy,
            "balanced_accuracy": video_metrics.balanced_accuracy,
            "macro_precision": video_metrics.macro_precision,
            "macro_recall": video_metrics.macro_recall,
            "macro_f1": video_metrics.macro_f1,
            "weighted_f1": video_metrics.weighted_f1,
            "macro_precision_cm": video_macro_precision,
            "macro_recall_cm": video_macro_recall,
            "per_class": video_metrics.per_class,
            "confusion_matrix": video_metrics.confusion_matrix.tolist(),
            "num_videos": int(len(video_res.video_ids)),
        },
        "classes": CLASSES,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results