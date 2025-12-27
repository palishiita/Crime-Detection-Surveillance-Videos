# src/evaluation/temporal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class VideoAggResult:
    y_true_video: np.ndarray
    y_pred_video: np.ndarray
    video_ids: List[str]


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def aggregate_video_predictions(
    logits: np.ndarray,
    y_true: np.ndarray,
    video_ids: List[str],
    frame_ids: List[int],
    method: str = "mean_probs",
    smoothing: str = "none",
    smoothing_alpha: float = 0.7,
) -> VideoAggResult:
    """
    Convert frame-level predictions into video-level predictions.

    Args:
      logits: (N, C) raw model outputs per frame
      y_true: (N,) true labels per frame
      video_ids: list length N, video id per frame
      frame_ids: list length N, frame index per frame (for sorting)
      method:
        - "majority_vote": vote on per-frame argmax labels
        - "mean_probs": average softmax probs over frames, then argmax
        - "max_probs": take max softmax probs over frames, then argmax
        - "topk_mean_probs": average top-k highest-confidence frames (per video)
      smoothing:
        - "none"
        - "ema_probs": exponential moving average over time (on probs), per video
      smoothing_alpha: for EMA, higher means more weight to current frame (0-1)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N,C). Got shape {logits.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (N,). Got shape {y_true.shape}")
    if len(video_ids) != logits.shape[0] or len(frame_ids) != logits.shape[0]:
        raise ValueError("video_ids/frame_ids length must match number of rows in logits.")

    probs = _softmax_np(logits)
    pred_frame = np.argmax(probs, axis=1)

    # Group indices by video_id
    by_vid: Dict[str, List[int]] = {}
    for i, vid in enumerate(video_ids):
        by_vid.setdefault(str(vid), []).append(i)

    y_true_video: List[int] = []
    y_pred_video: List[int] = []
    vids_out: List[str] = []

    for vid, idxs in by_vid.items():
        # sort frames by frame_id to keep temporal order
        idxs_sorted = sorted(idxs, key=lambda i: frame_ids[i])

        # Derive a single "true label" for video.
        # We assume all frames in a video share the same class folder label.
        # If not, we take majority of y_true.
        true_labels = y_true[idxs_sorted]
        true_video = int(np.bincount(true_labels).argmax())

        # Apply optional temporal smoothing on probs
        probs_seq = probs[idxs_sorted]  # (T, C)
        if smoothing == "ema_probs":
            smoothed = np.zeros_like(probs_seq)
            smoothed[0] = probs_seq[0]
            a = float(smoothing_alpha)
            for t in range(1, probs_seq.shape[0]):
                smoothed[t] = a * probs_seq[t] + (1.0 - a) * smoothed[t - 1]
            probs_seq = smoothed
        elif smoothing == "none":
            pass
        else:
            raise ValueError(f"Unknown smoothing: {smoothing}")

        # Aggregate
        if method == "majority_vote":
            votes = pred_frame[idxs_sorted]
            pred_video = int(np.bincount(votes).argmax())

        elif method == "mean_probs":
            pred_video = int(np.argmax(np.mean(probs_seq, axis=0)))

        elif method == "max_probs":
            pred_video = int(np.argmax(np.max(probs_seq, axis=0)))

        elif method.startswith("topk_mean_probs"):
            # Use only top-k frames by max confidence
            # method can be "topk_mean_probs" (defaults k=5) or "topk_mean_probs:k"
            k = 5
            if ":" in method:
                try:
                    k = int(method.split(":")[1])
                except Exception:
                    raise ValueError(f"Invalid topk method format: {method}")

            conf = np.max(probs_seq, axis=1)  # (T,)
            k = max(1, min(k, probs_seq.shape[0]))
            topk_idx = np.argsort(-conf)[:k]
            pred_video = int(np.argmax(np.mean(probs_seq[topk_idx], axis=0)))

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        y_true_video.append(true_video)
        y_pred_video.append(pred_video)
        vids_out.append(vid)

    return VideoAggResult(
        y_true_video=np.array(y_true_video, dtype=np.int64),
        y_pred_video=np.array(y_pred_video, dtype=np.int64),
        video_ids=vids_out,
    )