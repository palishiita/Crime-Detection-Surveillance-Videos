from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataloader import DataConfig, build_dataloaders
from src.data.dataset import CLASSES
from src.evaluation.plots import plot_history_csv
from src.models.model import build_mobilenetv2, build_resnet50, build_vgg16
from src.train.validate import validate
from src.train.validate_video import validate_video


@dataclass
class TrainConfig:
    # data
    data: DataConfig = field(default_factory=DataConfig)

    # model
    model_name: str = "resnet50"   # "resnet50" | "mobilenetv2" | "vgg16"
    num_classes: int = 6
    freeze_backbone: bool = False

    # optimization
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 10

    # regularization
    dropout: float = 0.4

    # training control
    device: str = "cpu"            # "cpu" or "cuda"
    log_every: int = 100

    # output
    out_dir: str = "experiments"
    experiment_name: str = "baseline"

    # early stopping (based on video-level balanced accuracy)
    early_stop_patience: int = 5

    # video-level validation config
    video_agg_method: str = "topk_mean_probs:20"
    video_smoothing: str = "ema_probs"
    video_smoothing_alpha: float = 0.7


def build_model(model_name: str, num_classes: int, freeze_backbone: bool, dropout: float) -> nn.Module:
    name = model_name.lower().strip()
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, dropout=dropout, freeze_backbone=freeze_backbone)
    if name in ("mobilenetv2", "mobilenet_v2", "mobilenet"):
        return build_mobilenetv2(num_classes=num_classes, dropout=0.3, freeze_backbone=freeze_backbone)
    if name == "vgg16":
        return build_vgg16(num_classes=num_classes, dropout=dropout, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model_name: {model_name}. Use: resnet50 | mobilenetv2 | vgg16")


def _ensure_dir(p: str | Path) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        x = batch[0].to(device)
        y = batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == y).sum().item()
        running_total += bs

        if step % log_every == 0 or step == 1:
            avg_loss = running_loss / max(running_total, 1)
            avg_acc = running_correct / max(running_total, 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return {"loss": float(epoch_loss), "accuracy": float(epoch_acc)}


def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    tag: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "tag": tag,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": CLASSES,
        },
        path,
    )


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Saved history: {path}")


def train(cfg: Optional[TrainConfig] = None) -> Dict:
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Data
    train_loader, val_loader, test_loader, artifacts = build_dataloaders(cfg.data)

    # Model
    model = build_model(cfg.model_name, cfg.num_classes, cfg.freeze_backbone, cfg.dropout).to(device)

    # Loss strategy:
    # - If you are using a balanced subset (max_per_class_train), use plain CE.
    # - Otherwise, use class-weighted CE for imbalance.
    if getattr(cfg.data, "max_per_class_train", None) is not None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        class_weights = artifacts["class_weights"].to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler (monitor VIDEO balanced accuracy)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Output dirs
    exp_dir = Path(cfg.out_dir) / cfg.model_name / cfg.experiment_name
    ckpt_dir = Path(_ensure_dir(exp_dir / "checkpoints"))
    log_dir = Path(_ensure_dir(exp_dir / "logs"))
    plot_dir = Path(_ensure_dir(exp_dir / "plots"))

    best_video_bal_acc = float("-inf")
    best_video_macro_f1 = float("-inf")
    best_bal_epoch = -1
    best_f1_epoch = -1

    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print(f"Device: {device}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_every=cfg.log_every,
        )

        # Frame-level validation (still useful to log)
        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            cm_normalize=None,
        )

        # Video-level validation (main monitor)
        val_video_stats = validate_video(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            cm_normalize=None,
            agg_method=cfg.video_agg_method,
            smoothing=cfg.video_smoothing,
            smoothing_alpha=cfg.video_smoothing_alpha,
        )

        frame_bal_acc = float(val_stats["balanced_accuracy"])
        frame_macro_f1 = float(val_stats["macro_f1"])

        video_bal_acc = float(val_video_stats["balanced_accuracy"])
        video_macro_f1 = float(val_video_stats["macro_f1"])

        scheduler.step(video_bal_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],

            "val_loss": val_stats["loss"] if val_stats["loss"] is not None else float("nan"),
            "val_acc": float(val_stats["accuracy"]),
            "val_balanced_acc": frame_bal_acc,
            "val_macro_f1": frame_macro_f1,
            "val_weighted_f1": float(val_stats["weighted_f1"]),

            "val_video_loss": val_video_stats["loss"] if val_video_stats["loss"] is not None else float("nan"),
            "val_video_acc": float(val_video_stats["accuracy"]),
            "val_video_balanced_acc": video_bal_acc,
            "val_video_macro_f1": video_macro_f1,
            "val_video_weighted_f1": float(val_video_stats["weighted_f1"]),
            "val_num_videos": int(val_video_stats.get("num_videos", -1)),

            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"Train loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | "
            f"Val(frame) bal_acc={row['val_balanced_acc']:.4f} macro_f1={row['val_macro_f1']:.4f} | "
            f"Val(video) bal_acc={row['val_video_balanced_acc']:.4f} macro_f1={row['val_video_macro_f1']:.4f} "
            f"(videos={row['val_num_videos']})"
        )

        # Save best by VIDEO balanced accuracy
        if video_bal_acc > best_video_bal_acc:
            best_video_bal_acc = video_bal_acc
            best_bal_epoch = epoch
            _save_checkpoint(
                path=str(ckpt_dir / "best_video_bal_acc.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_video_bal_acc,
                tag="best_video_balanced_accuracy",
            )
            print(
                f"Saved best_video_bal_acc.pt (video_balanced_accuracy={best_video_bal_acc:.4f}, epoch={best_bal_epoch})"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save best by VIDEO macro F1
        if video_macro_f1 > best_video_macro_f1:
            best_video_macro_f1 = video_macro_f1
            best_f1_epoch = epoch
            _save_checkpoint(
                path=str(ckpt_dir / "best_video_macro_f1.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_video_macro_f1,
                tag="best_video_macro_f1",
            )
            print(
                f"Saved best_video_macro_f1.pt (video_macro_f1={best_video_macro_f1:.4f}, epoch={best_f1_epoch})"
            )

        # Early stopping based on VIDEO balanced accuracy improvements
        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. Best video balanced accuracy at epoch {best_bal_epoch}"
            )
            break

    hist_path = str(log_dir / "history.csv")
    _write_csv(history, hist_path)

    plot_paths = plot_history_csv(hist_path, plot_dir)

    return {
        "best_video_balanced_accuracy": best_video_bal_acc,
        "best_video_balanced_accuracy_epoch": best_bal_epoch,
        "best_video_macro_f1": best_video_macro_f1,
        "best_video_macro_f1_epoch": best_f1_epoch,
        "checkpoint_best_video_bal_acc": str(ckpt_dir / "best_video_bal_acc.pt"),
        "checkpoint_best_video_macro_f1": str(ckpt_dir / "best_video_macro_f1.pt"),
        "history_csv": hist_path,
        "plots": plot_paths,
        "experiment_dir": str(exp_dir),
    }