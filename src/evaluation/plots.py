from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def _read_history_csv(path: Union[str, Path]) -> List[Dict[str, float]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History CSV not found: {path}")

    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert values that look like numbers into floats where possible
            parsed: Dict[str, float] = {}
            for k, v in r.items():
                if v is None:
                    continue
                v = v.strip()
                try:
                    parsed[k] = float(v)
                except ValueError:
                    # Keep non-numeric as-is (rare for our history.csv)
                    pass
            rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No rows found in history CSV: {path}")

    return rows


def plot_history_csv(
    history_csv: Union[str, Path],
    out_dir: Union[str, Path],
    prefix: str = "",
) -> Dict[str, str]:
    """
    Creates interpretable training plots from history.csv.

    Produces:
      - loss_curve.png (train_loss & val_loss)
      - val_balanced_accuracy.png
      - val_macro_f1.png

    Returns:
      dict of plot_name -> file_path
    """
    rows = _read_history_csv(history_csv)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]

    def series(key: str) -> List[float]:
        return [float(r.get(key, float("nan"))) for r in rows]

    train_loss = series("train_loss")
    val_loss = series("val_loss")
    val_bal_acc = series("val_balanced_acc")
    val_macro_f1 = series("val_macro_f1")

    saved: Dict[str, str] = {}

    # Plot 1: Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_path = out_dir / f"{prefix}loss_curve.png"
    plt.savefig(loss_path, bbox_inches="tight", dpi=200)
    plt.close()
    saved["loss_curve"] = str(loss_path)

    # Plot 2: Balanced accuracy
    plt.figure()
    plt.plot(epochs, val_bal_acc, label="val_balanced_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Validation Balanced Accuracy")
    plt.legend()
    bal_path = out_dir / f"{prefix}val_balanced_accuracy.png"
    plt.savefig(bal_path, bbox_inches="tight", dpi=200)
    plt.close()
    saved["val_balanced_accuracy"] = str(bal_path)

    # Plot 3: Macro F1
    plt.figure()
    plt.plot(epochs, val_macro_f1, label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1")
    plt.legend()
    f1_path = out_dir / f"{prefix}val_macro_f1.png"
    plt.savefig(f1_path, bbox_inches="tight", dpi=200)
    plt.close()
    saved["val_macro_f1"] = str(f1_path)

    return saved
