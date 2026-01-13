# src/train/validate_video.py
from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CLASSES
from src.evaluation.metrics import compute_metrics
from src.evaluation.temporal import aggregate_video_predictions


@torch.no_grad()
def validate_video(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    cm_normalize: Optional[str] = None,
    agg_method: str = "mean_probs",
    smoothing: str = "none",
    smoothing_alpha: float = 0.7,
) -> Dict:
    """
    Video-level validation using temporal aggregation.

    Loader must yield:
      (x, y, meta)
    where meta is a list of SampleMeta with .video_id and .frame_id
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    logits_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    video_ids: List[str] = []
    frame_ids: List[int] = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        metas = batch[2]
        for m in metas:
            video_ids.append(str(m.video_id))
            frame_ids.append(int(m.frame_id))

        logits = model(x)

        if loss_fn is not None:
            loss = loss_fn(logits, y)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

        logits_all.append(logits.detach().cpu().numpy())
        y_true_all.append(y.detach().cpu().numpy())

    avg_loss = None
    if loss_fn is not None and total_samples > 0:
        avg_loss = total_loss / total_samples

    logits_np = np.concatenate(logits_all, axis=0)
    y_true_np = np.concatenate(y_true_all, axis=0)

    video_res = aggregate_video_predictions(
        logits=logits_np,
        y_true=y_true_np,
        video_ids=video_ids,
        frame_ids=frame_ids,
        method=agg_method,
        smoothing=smoothing,
        smoothing_alpha=smoothing_alpha,
    )

    metrics = compute_metrics(
        y_true=video_res.y_true_video,
        y_pred=video_res.y_pred_video,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize=cm_normalize,
    )

    return {
        "loss": avg_loss,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "per_class": metrics.per_class,
        "confusion_matrix": metrics.confusion_matrix,
        "num_videos": int(len(video_res.video_ids)),
    }