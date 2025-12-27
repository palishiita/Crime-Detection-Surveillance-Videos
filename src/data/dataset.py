from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset


CLASSES: List[str] = ["Abuse", "Assault", "Fighting", "Normal_Videos", "Robbery", "Shooting"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


@dataclass(frozen=True)
class SampleMeta:
    """Extra info we can keep for debugging / video-wise evaluation later."""
    path: str
    label_name: str
    label: int
    video_id: str
    frame_id: int
    filename: str


def _parse_video_and_frame(filename: str) -> Tuple[str, int]:
    """
    Parse video_id and frame_id from filenames like:
      Normal_Videos_003_x264_0.png

    Returns:
      video_id (string, keeps leading zeros)
      frame_id (int)
    """
    stem = Path(filename).stem  # remove extension

    # Common pattern: <anything>_<video>_x264_<frame>
    m = re.search(r"_(\d+)_x264_(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))

    # Fallback: take last two numeric groups in the filename
    nums = re.findall(r"\d+", stem)
    if len(nums) >= 2:
        return nums[-2], int(nums[-1])

    raise ValueError(f"Could not parse video_id/frame_id from filename: {filename}")


class CrimeFramesDataset(Dataset):
    """
    Frame-level dataset for crime classification.
    Expects directory layout like:

      dataset/
        train/
          Abuse/*.png
          ...
          Normal_Videos/*.png
        test/
          Abuse/*.png
          ...

    Each sample corresponds to a single image frame.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str,  # "train" or "test"
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
        strict_class_folders: bool = True,
        max_per_class: Optional[int] = None,  # limit samples per class (debug/balanced runs)
        seed: int = 42,                       # reproducible sampling
    ) -> None:
        """
        Args:
          root_dir: dataset directory (e.g., "dataset")
          split: "train" or "test"
          classes: class list; defaults to global CLASSES
          transform: torchvision transform pipeline
          return_meta: if True, __getitem__ returns (img, label, meta)
          strict_class_folders: if True, raises if class folders are missing
          max_per_class: if set, randomly selects up to N frames per class (balanced subset)
          seed: RNG seed used when max_per_class is set (and also to shuffle per class)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta

        self.classes = list(classes) if classes is not None else list(CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        missing = [c for c in self.classes if not (self.split_dir / c).exists()]
        if missing and strict_class_folders:
            raise FileNotFoundError(
                f"Missing class folders in {self.split_dir}: {missing}\n"
                f"Expected folders: {self.classes}"
            )

        self.samples: List[Tuple[Path, int, str, int]] = []
        # Each tuple: (path, label_idx, video_id, frame_id)

        rng = random.Random(seed)

        # Optional info message (helps avoid confusion later)
        if max_per_class is not None:
            print(f"[CrimeFramesDataset] Using max_per_class={max_per_class} for split='{split}' (seed={seed}).")

        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue  # only happens if strict_class_folders=False

            files = [
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS
            ]

            # shuffle so subset isn't always the same first-N
            rng.shuffle(files)

            if max_per_class is not None:
                files = files[:max_per_class]

            label = self.class_to_idx[class_name]
            for p in files:
                video_id, frame_id = _parse_video_and_frame(p.name)
                self.samples.append((p, label, video_id, frame_id))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.split_dir}. "
                f"Check folder names and supported extensions: {sorted(self.IMG_EXTS)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, video_id, frame_id = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if not self.return_meta:
            return img, label

        meta = SampleMeta(
            path=str(path),
            label_name=self.classes[label],
            label=label,
            video_id=video_id,
            frame_id=frame_id,
            filename=path.name,
        )
        return img, label, meta

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for _, label, _, _ in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def group_by_video(self) -> Dict[str, List[int]]:
        """
        Returns mapping: video_id -> list of sample indices.
        Useful later for video-level evaluation/aggregation.
        """
        out: Dict[str, List[int]] = {}
        for i, (_, _, video_id, _) in enumerate(self.samples):
            out.setdefault(video_id, []).append(i)
        return out