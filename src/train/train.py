from __future__ import annotations
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
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
    model_name: str = "resnet50"
    num_classes: int = 6
    # transfer learning / fine-tuning
    freeze_backbone: bool = True           
    fine_tune: bool = True                
    fine_tune_start_epoch: int = 4 
    backbone_lr: float = 1e-5 
    head_lr: float = 1e-4 
    freeze_bn_stats: bool = True
    # optimization
    weight_decay: float = 1e-4
    epochs: int = 10
    # regularization
    dropout: float = 0.4
    # training control
    device: str = "cpu"
    log_every: int = 100
    # output
    out_dir: str = "experiments"
    experiment_name: str = "baseline"
    # early stopping
    early_stop_patience: int = 5
    monitor_metric: str = "video_macro_recall"
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


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _freeze_all(model: nn.Module) -> None:
    _set_requires_grad(model, False)


def _get_head_module(model: nn.Module, model_name: str) -> nn.Module:
    name = model_name.lower().strip()
    if name == "resnet50":
        return model.fc
    # mobilenetv2 and vgg16 both use .classifier
    return model.classifier


def _freeze_backbone_keep_head_trainable(model: nn.Module, model_name: str) -> None:
    _freeze_all(model)
    head = _get_head_module(model, model_name)
    _set_requires_grad(head, True)


def _set_batchnorm_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _unfreeze_last_blocks(model: nn.Module, model_name: str) -> List[nn.Module]:
    name = model_name.lower().strip()
    unfrozen: List[nn.Module] = []

    if name == "resnet50":
        # Unfreeze layer4 (best default)
        _set_requires_grad(model.layer4, True)
        unfrozen.append(model.layer4)

    elif name in ("mobilenetv2", "mobilenet_v2", "mobilenet"):
        # Unfreeze last block 
        if hasattr(model, "features"):
            _set_requires_grad(model.features[-1], True)
            unfrozen.append(model.features[-1])

    elif name == "vgg16":
        # Unfreeze last conv block
        if hasattr(model, "features"):
            # VGG features is a Sequential; unfreeze last chunk
            last_n = 5
            for layer in list(model.features.children())[-last_n:]:
                _set_requires_grad(layer, True)
                unfrozen.append(layer)

    else:
        raise ValueError(f"Unknown model_name for unfreeze: {model_name}")

    return unfrozen


def _build_optimizer(
    model: nn.Module,
    model_name: str,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Discriminative LR: head gets higher LR, backbone gets lower LR.
    Uses id-based filtering to avoid tensor equality comparisons.
    """
    head = _get_head_module(model, model_name)

    head_params = [p for p in head.parameters() if p.requires_grad]
    head_param_ids = {id(p) for p in head_params}

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and (id(p) not in head_param_ids)
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check freeze/unfreeze logic.")

    return torch.optim.Adam(param_groups, weight_decay=weight_decay)

# Training loop
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
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"Saved history: {path}")


def _get_monitor_value(cfg: TrainConfig, val_video_stats: Dict[str, float]) -> Tuple[str, float]:
    if cfg.monitor_metric == "video_macro_recall":
        if "macro_recall" in val_video_stats:
            return ("video_macro_recall", float(val_video_stats["macro_recall"]))
        return ("video_macro_f1", float(val_video_stats.get("macro_f1", float("-inf"))))
    if cfg.monitor_metric == "video_macro_f1":
        return ("video_macro_f1", float(val_video_stats.get("macro_f1", float("-inf"))))
    return ("video_balanced_accuracy", float(val_video_stats.get("balanced_accuracy", float("-inf"))))


def train(cfg: Optional[TrainConfig] = None) -> Dict:
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    # Data
    train_loader, val_loader, test_loader, artifacts = build_dataloaders(cfg.data)
    weighted_sampling = bool(getattr(cfg.data, "weighted_sampling", False))

    if getattr(cfg.data, "max_per_class_train", None) is not None:
        # balanced subset => plain CE
        loss_fn = nn.CrossEntropyLoss()
    else:
        if weighted_sampling:
            loss_fn = nn.CrossEntropyLoss()
        else:
            class_weights = artifacts["class_weights"].to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = build_model(cfg.model_name, cfg.num_classes, cfg.freeze_backbone, cfg.dropout).to(device)

    if cfg.freeze_backbone:
        _freeze_backbone_keep_head_trainable(model, cfg.model_name)

    optimizer = _build_optimizer(
        model=model,
        model_name=cfg.model_name,
        head_lr=cfg.head_lr,
        backbone_lr=cfg.backbone_lr,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler (monitor chosen video metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    exp_dir = Path(cfg.out_dir) / cfg.model_name / cfg.experiment_name
    ckpt_dir = Path(_ensure_dir(exp_dir / "checkpoints"))
    log_dir = Path(_ensure_dir(exp_dir / "logs"))
    plot_dir = Path(_ensure_dir(exp_dir / "plots"))

    best_score = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict] = []

    # Track extra baselines too
    best_video_bal_acc = float("-inf")
    best_video_macro_f1 = float("-inf")

    did_unfreeze = False

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print(f"Device: {device}")

        if cfg.fine_tune and (epoch >= cfg.fine_tune_start_epoch) and not did_unfreeze:
            print(f"Fine-tuning: unfreezing last block(s) for {cfg.model_name} (epoch {epoch})")
            _unfreeze_last_blocks(model, cfg.model_name)

            if cfg.freeze_bn_stats:
                _set_batchnorm_eval(model)

            optimizer = _build_optimizer(
                model=model,
                model_name=cfg.model_name,
                head_lr=cfg.head_lr,
                backbone_lr=cfg.backbone_lr,
                weight_decay=cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )
            did_unfreeze = True

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_every=cfg.log_every,
        )

        # Frame-level validation (logging)
        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            cm_normalize=None,
        )

        # Video-level validation (primary)
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

        monitor_name, monitor_score = _get_monitor_value(cfg, val_video_stats)
        scheduler.step(monitor_score)

        # log row
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

            "monitor_name": monitor_name,
            "monitor_score": float(monitor_score),

            "lr_group0": float(optimizer.param_groups[0]["lr"]),
            "lr_group1": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else float("nan"),
            "did_unfreeze": int(did_unfreeze),
        }
        if "macro_recall" in val_video_stats:
            row["val_video_macro_recall"] = float(val_video_stats["macro_recall"])

        history.append(row)

        print(
            f"Train loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | "
            f"Val(frame) bal_acc={row['val_balanced_acc']:.4f} macro_f1={row['val_macro_f1']:.4f} | "
            f"Val(video) bal_acc={row['val_video_balanced_acc']:.4f} macro_f1={row['val_video_macro_f1']:.4f} "
            f"{'(macro_recall=' + str(row.get('val_video_macro_recall')) + ')' if 'val_video_macro_recall' in row else ''} "
            f"(videos={row['val_num_videos']}) | Monitor={monitor_name}:{monitor_score:.4f}"
        )

        if monitor_score > best_score:
            best_score = monitor_score
            best_epoch = epoch
            _save_checkpoint(
                path=str(ckpt_dir / "best_monitor.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                tag=f"best_{monitor_name}",
            )
            print(f"Saved best_monitor.pt ({monitor_name}={best_score:.4f}, epoch={best_epoch})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if video_bal_acc > best_video_bal_acc:
            best_video_bal_acc = video_bal_acc
            _save_checkpoint(
                path=str(ckpt_dir / "best_video_bal_acc.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_video_bal_acc,
                tag="best_video_balanced_accuracy",
            )

        if video_macro_f1 > best_video_macro_f1:
            best_video_macro_f1 = video_macro_f1
            _save_checkpoint(
                path=str(ckpt_dir / "best_video_macro_f1.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_video_macro_f1,
                tag="best_video_macro_f1",
            )

        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best {monitor_name} at epoch {best_epoch}")
            break

    hist_path = str(log_dir / "history.csv")
    _write_csv(history, hist_path)

    plot_paths = plot_history_csv(hist_path, plot_dir)

    return {
        "best_monitor_score": best_score,
        "best_monitor_epoch": best_epoch,
        "monitor_metric": cfg.monitor_metric,
        "best_video_balanced_accuracy": best_video_bal_acc,
        "best_video_macro_f1": best_video_macro_f1,
        "checkpoint_best_monitor": str(ckpt_dir / "best_monitor.pt"),
        "checkpoint_best_video_bal_acc": str(ckpt_dir / "best_video_bal_acc.pt"),
        "checkpoint_best_video_macro_f1": str(ckpt_dir / "best_video_macro_f1.pt"),
        "history_csv": hist_path,
        "plots": plot_paths,
        "experiment_dir": str(exp_dir),
    }
