from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from src.data.dataset import CrimeFramesDataset, CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    root_dir: Union[str, Path] = "dataset"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = False

    # imbalance handling (train only)
    weighted_sampling: bool = True

    # folder/metadata behavior
    strict_class_folders: bool = True
    return_meta: bool = False
    seed: int = 42

    # debug caps (optional) - caps frames per class
    max_per_class_train: Optional[int] = None
    max_per_class_test: Optional[int] = None

    # validation split from train
    val_split: float = 0.1
    video_wise_split: bool = True  # if False -> frame-wise split


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tfms, test_tfms


def collate_with_meta(batch):
    """
    Dataset item format:
      - (img, label)
      - or (img, label, meta)

    Returns:
      - imgs: Tensor [B,C,H,W]
      - labels: Tensor [B]
      - metas: list (kept as python list, not collated into tensors)
    """
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

    if len(batch[0]) >= 3:
        metas = [b[2] for b in batch]
        return imgs, labels, metas

    return imgs, labels


def _make_weighted_sampler(labels: List[int], num_classes: int, seed: int = 42) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)

    class_weights = 1.0 / counts
    sample_weights = class_weights[np.array(labels, dtype=np.int64)]

    g = torch.Generator()
    g.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )


def _compute_class_weights_from_labels(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def _counts_by_class_from_labels(labels: List[int], classes: List[str]) -> Dict[str, int]:
    num_classes = len(classes)
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes)
    return {classes[i]: int(counts[i]) for i in range(num_classes)}


def _split_train_indices_frame_wise(
    train_ds: CrimeFramesDataset,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1). Got: {val_split}")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(train_ds))
    rng.shuffle(all_idx)

    n_val = int(round(len(all_idx) * val_split))
    n_val = max(1, n_val)

    val_idx = all_idx[:n_val].tolist()
    train_idx = all_idx[n_val:].tolist()
    return train_idx, val_idx


def _split_train_indices_video_wise_stratified(
    train_ds: CrimeFramesDataset,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Stratified video-wise split:
      - Keep all frames of a video together
      - Approximate class stratification by splitting videos within each class

    Assumes train_ds.samples items are:
      (path, label, video_id, frame_id)
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1). Got: {val_split}")

    rng = np.random.RandomState(seed)

    # label -> video_id -> list of indices
    by_label_video = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(train_ds.samples):
        label = int(sample[1])
        video_id = str(sample[2])
        by_label_video[label][video_id].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label, video_map in by_label_video.items():
        videos = list(video_map.keys())
        rng.shuffle(videos)

        # ensure at least 1 video stays in train if possible
        if len(videos) <= 1:
            val_videos = set()
        else:
            n_val = int(round(len(videos) * val_split))
            n_val = max(1, min(n_val, len(videos) - 1))
            val_videos = set(videos[:n_val])

        for vid, idxs in video_map.items():
            if vid in val_videos:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_dataloaders(
    cfg: Optional[DataConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns:
      train_loader, val_loader, test_loader, artifacts
    """
    if cfg is None:
        cfg = DataConfig()

    train_tfms, test_tfms = build_transforms(cfg.img_size)

    # Build a SINGLE train dataset, then split indices into train/val
    train_full_ds = CrimeFramesDataset(
        root_dir=cfg.root_dir,
        split="train",
        classes=list(CLASSES),
        transform=train_tfms,
        return_meta=cfg.return_meta,
        strict_class_folders=cfg.strict_class_folders,
        max_per_class=cfg.max_per_class_train,
        seed=cfg.seed,
    )

    test_ds = CrimeFramesDataset(
        root_dir=cfg.root_dir,
        split="test",
        classes=list(CLASSES),
        transform=test_tfms,
        return_meta=cfg.return_meta,
        strict_class_folders=cfg.strict_class_folders,
        max_per_class=cfg.max_per_class_test,
        seed=cfg.seed,
    )

    # Split train -> train/val
    if cfg.val_split is None or cfg.val_split <= 0.0:
        train_idx = list(range(len(train_full_ds)))
        val_idx: List[int] = []
    else:
        if cfg.video_wise_split:
            train_idx, val_idx = _split_train_indices_video_wise_stratified(train_full_ds, cfg.val_split, cfg.seed)
        else:
            train_idx, val_idx = _split_train_indices_frame_wise(train_full_ds, cfg.val_split, cfg.seed)

    train_ds = Subset(train_full_ds, train_idx)
    val_ds = Subset(train_full_ds, val_idx)

    # Train sampler (train only)
    sampler = None
    shuffle = True
    if cfg.weighted_sampling and len(train_idx) > 0:
        train_labels = [int(train_full_ds.samples[i][1]) for i in train_idx]
        sampler = _make_weighted_sampler(train_labels, num_classes=len(train_full_ds.classes), seed=cfg.seed)
        shuffle = False

    # collate function when meta is enabled
    collate = collate_with_meta if cfg.return_meta else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
    )

    # Artifacts
    train_labels = [int(train_full_ds.samples[i][1]) for i in train_idx]
    val_labels = [int(train_full_ds.samples[i][1]) for i in val_idx]
    test_labels = [int(lbl) for (_, lbl, _, _) in test_ds.samples]

    artifacts: Dict = {}
    artifacts["classes"] = train_full_ds.classes
    artifacts["class_weights"] = _compute_class_weights_from_labels(train_labels, num_classes=len(train_full_ds.classes))
    artifacts["train_counts"] = _counts_by_class_from_labels(train_labels, train_full_ds.classes)
    artifacts["val_counts"] = _counts_by_class_from_labels(val_labels, train_full_ds.classes)
    artifacts["test_counts"] = _counts_by_class_from_labels(test_labels, train_full_ds.classes)
    artifacts["split_sizes"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_ds)}

    return train_loader, val_loader, test_loader, artifacts


def sanity_check(cfg: Optional[DataConfig] = None, num_batches: int = 1) -> None:
    train_loader, val_loader, test_loader, artifacts = build_dataloaders(cfg)

    print("CLASSES:", artifacts["classes"])
    print("Split sizes:", artifacts["split_sizes"])
    print("Train counts:", artifacts["train_counts"])
    print("Val counts:", artifacts["val_counts"])
    print("Test counts:", artifacts["test_counts"])
    print("Class weights (loss):", artifacts["class_weights"].tolist())

    for i, batch in enumerate(train_loader):
        x, y = batch[0], batch[1]
        print(f"Train batch {i}: x={tuple(x.shape)}, y={tuple(y.shape)}")
        if i + 1 >= num_batches:
            break

    for i, batch in enumerate(val_loader):
        x, y = batch[0], batch[1]
        print(f"Val batch {i}: x={tuple(x.shape)}, y={tuple(y.shape)}")
        break