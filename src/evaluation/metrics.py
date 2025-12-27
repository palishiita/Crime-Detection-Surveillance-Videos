# src/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class MetricsResult:
    """Container for common classification metrics."""
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray


def _to_numpy(x) -> np.ndarray:
    """Accepts torch tensors, lists, numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_confusion_matrix(
    y_true,
    y_pred,
    num_classes: Optional[int] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Args:
      y_true, y_pred: array-like class indices
      num_classes: fixes matrix shape to (C,C)
      normalize: None | 'true' | 'pred' | 'all'  (sklearn style)

    Returns:
      confusion matrix (C x C)
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)

    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def compute_metrics(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    cm_normalize: Optional[str] = None,
    zero_division: int = 0,
) -> MetricsResult:
    """
    Compute metrics for multiclass classification.

    Args:
      y_true: ground-truth class indices (N,)
      y_pred: predicted class indices (N,)
      class_names: list of class names (length C). If None, uses string indices.
      num_classes: number of classes. If None, inferred from data.
      cm_normalize: None | 'true' | 'pred' | 'all' (normalization for confusion matrix)
      zero_division: how sklearn handles division by zero in precision/recall/F1

    Returns:
      MetricsResult dataclass
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        assert len(class_names) == num_classes, (
            f"class_names length ({len(class_names)}) must match num_classes ({num_classes})"
        )

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Macro + weighted
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=zero_division
    )

    # Per-class (using sklearn classification_report for convenience)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=zero_division,
    )

    # Convert report to a simpler per-class dict
    per_class: Dict[str, Dict[str, float]] = {}
    for cname in class_names:
        # keys: 'precision', 'recall', 'f1-score', 'support'
        per_class[cname] = {
            "precision": float(report[cname]["precision"]),
            "recall": float(report[cname]["recall"]),
            "f1": float(report[cname]["f1-score"]),
            "support": float(report[cname]["support"]),
        }

    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        normalize=cm_normalize,
    )

    return MetricsResult(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_precision=float(w_p),
        weighted_recall=float(w_r),
        weighted_f1=float(w_f1),
        per_class=per_class,
        confusion_matrix=cm,
    )


def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits (N, C) -> predicted class indices (N,).
    """
    return torch.argmax(logits, dim=1)


def collect_preds_from_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on a dataloader and collects (y_true, y_pred) arrays.

    Assumes loader yields:
      (x, y) or (x, y, meta)

    Returns:
      y_true_np, y_pred_np
    """
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            logits = model(x)
            preds = logits_to_preds(logits)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true_all, axis=0)
    y_pred_np = np.concatenate(y_pred_all, axis=0)
    return y_true_np, y_pred_np
