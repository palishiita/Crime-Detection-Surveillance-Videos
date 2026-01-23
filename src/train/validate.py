from __future__ import annotations
import torch
from typing import Dict, Optional
from src.data.dataset import CLASSES
from torch.utils.data import DataLoader
from src.evaluation.metrics import collect_preds_from_loader, compute_metrics


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    cm_normalize: Optional[str] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    if loss_fn is not None:
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                y = batch[1].to(device)

                logits = model(x)
                loss = loss_fn(logits, y)

                bs = y.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
    else:
        avg_loss = None

    y_true, y_pred = collect_preds_from_loader(model, loader, device)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASSES,
        num_classes=len(CLASSES),
        cm_normalize=cm_normalize,
    )

    results = {
        "loss": avg_loss,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "per_class": metrics.per_class,
        "confusion_matrix": metrics.confusion_matrix,
    }

    return results