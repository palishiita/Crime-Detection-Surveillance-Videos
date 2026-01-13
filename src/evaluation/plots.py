from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Union, Any

import matplotlib.pyplot as plt


def _read_history_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History CSV not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed: Dict[str, Any] = {}
            for k, v in r.items():
                if v is None:
                    parsed[k] = None
                    continue
                v = v.strip()
                if v == "":
                    parsed[k] = None
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No rows found in history CSV: {path}")

    return rows


def _has_key(rows: List[Dict[str, Any]], key: str) -> bool:
    for r in rows:
        if key in r and r[key] is not None:
            return True
    return False


def plot_history_csv(
    history_csv: Union[str, Path],
    out_dir: Union[str, Path],
    prefix: str = "",
) -> Dict[str, str]:
    """
    Creates interpretable training plots from history.csv.

    Backward compatible:
      - If only frame-level keys exist -> plots frame curves.
      - If video-level keys exist -> plots video curves too.

    Common expected columns (frame):
      train_loss, val_loss, val_balanced_acc, val_macro_f1

    New optional columns (video):
      val_video_loss, val_video_balanced_acc, val_video_macro_f1
    """
    rows = _read_history_csv(history_csv)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    epochs = [int(r.get("epoch", i + 1) or (i + 1)) for i, r in enumerate(rows)]

    def series(key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = r.get(key, None)
            out.append(float(v) if v is not None else float("nan"))
        return out

    saved: Dict[str, str] = {}

    # Detect availability
    has_val_loss = _has_key(rows, "val_loss")
    has_val_video_loss = _has_key(rows, "val_video_loss")

    has_val_bal = _has_key(rows, "val_balanced_acc")
    has_val_video_bal = _has_key(rows, "val_video_balanced_acc")

    has_val_f1 = _has_key(rows, "val_macro_f1")
    has_val_video_f1 = _has_key(rows, "val_video_macro_f1")

    # -----------------------
    # Plot 1: Loss curves
    # -----------------------
    train_loss = series("train_loss")

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")

    if has_val_loss:
        plt.plot(epochs, series("val_loss"), label="val_loss")

    if has_val_video_loss:
        plt.plot(epochs, series("val_video_loss"), label="val_video_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    loss_path = out_dir / f"{prefix}loss_curve.png"
    plt.savefig(loss_path, bbox_inches="tight", dpi=200)
    plt.close()
    saved["loss_curve"] = str(loss_path)

    # ------------------------------------
    # Plot 2: Balanced Accuracy curves
    # ------------------------------------
    if has_val_bal or has_val_video_bal:
        plt.figure()

        if has_val_bal:
            plt.plot(epochs, series("val_balanced_acc"), label="val_balanced_acc")

        if has_val_video_bal:
            plt.plot(epochs, series("val_video_balanced_acc"), label="val_video_balanced_acc")

        plt.xlabel("Epoch")
        plt.ylabel("Balanced Accuracy")
        plt.title("Validation Balanced Accuracy")
        plt.legend()

        bal_path = out_dir / f"{prefix}val_balanced_accuracy.png"
        plt.savefig(bal_path, bbox_inches="tight", dpi=200)
        plt.close()
        saved["val_balanced_accuracy"] = str(bal_path)

    # -------------------------
    # Plot 3: Macro F1 curves
    # -------------------------
    if has_val_f1 or has_val_video_f1:
        plt.figure()

        if has_val_f1:
            plt.plot(epochs, series("val_macro_f1"), label="val_macro_f1")

        if has_val_video_f1:
            plt.plot(epochs, series("val_video_macro_f1"), label="val_video_macro_f1")

        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Validation Macro F1")
        plt.legend()

        f1_path = out_dir / f"{prefix}val_macro_f1.png"
        plt.savefig(f1_path, bbox_inches="tight", dpi=200)
        plt.close()
        saved["val_macro_f1"] = str(f1_path)

    return saved