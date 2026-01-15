from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> Dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cols = {h: [] for h in header}
        for line in f:
            parts = line.strip().split(",")
            for h, v in zip(header, parts):
                # best-effort numeric parse
                try:
                    cols[h].append(float(v))
                except Exception:
                    cols[h].append(v)
    return cols


def _plot_series(xs, ys, title: str, xlabel: str, ylabel: str, out_path: str) -> str:
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_history_csv(history_csv: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    cols = _read_csv(history_csv)

    epoch = cols.get("epoch")
    if epoch is None:
        raise ValueError("history.csv must contain an 'epoch' column")

    out_paths: Dict[str, str] = {}

    # Loss
    if "train_loss" in cols:
        out_paths["train_loss"] = _plot_series(
            epoch, cols["train_loss"],
            "Train Loss", "Epoch", "Loss",
            os.path.join(out_dir, "train_loss.png"),
        )

    if "val_loss" in cols:
        out_paths["val_loss"] = _plot_series(
            epoch, cols["val_loss"],
            "Val Loss (Frame)", "Epoch", "Loss",
            os.path.join(out_dir, "val_loss.png"),
        )

    # Balanced accuracy (frame/video)
    if "val_balanced_acc" in cols:
        out_paths["val_balanced_acc"] = _plot_series(
            epoch, cols["val_balanced_acc"],
            "Val Balanced Accuracy (Frame)", "Epoch", "Balanced Accuracy",
            os.path.join(out_dir, "val_balanced_acc.png"),
        )

    if "val_video_balanced_acc" in cols:
        out_paths["val_video_balanced_acc"] = _plot_series(
            epoch, cols["val_video_balanced_acc"],
            "Val Balanced Accuracy (Video)", "Epoch", "Balanced Accuracy",
            os.path.join(out_dir, "val_video_balanced_acc.png"),
        )

    # Macro F1 (frame/video)
    if "val_macro_f1" in cols:
        out_paths["val_macro_f1"] = _plot_series(
            epoch, cols["val_macro_f1"],
            "Val Macro F1 (Frame)", "Epoch", "Macro F1",
            os.path.join(out_dir, "val_macro_f1.png"),
        )

    if "val_video_macro_f1" in cols:
        out_paths["val_video_macro_f1"] = _plot_series(
            epoch, cols["val_video_macro_f1"],
            "Val Macro F1 (Video)", "Epoch", "Macro F1",
            os.path.join(out_dir, "val_video_macro_f1.png"),
        )

    # Macro Recall (video) — NEW (only if present)
    if "val_video_macro_recall" in cols:
        out_paths["val_video_macro_recall"] = _plot_series(
            epoch, cols["val_video_macro_recall"],
            "Val Macro Recall (Video)", "Epoch", "Macro Recall",
            os.path.join(out_dir, "val_video_macro_recall.png"),
        )

    return out_paths