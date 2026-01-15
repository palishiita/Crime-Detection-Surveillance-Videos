from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

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


def _parse_topk(method: str, default_k: Union[int, float] = 0.05) -> Union[int, float]:
    """
    Parses:
      - "topk_mean_probs" -> default_k
      - "topk_mean_probs:20" -> int k
      - "topk_mean_probs:0.05" -> float fraction k
    """
    if ":" not in method:
        return default_k
    raw = method.split(":", 1)[1].strip()
    if raw == "":
        return default_k
    try:
        if "." in raw:
            kf = float(raw)
            return kf
        return int(raw)
    except Exception as e:
        raise ValueError(f"Invalid topk method format: {method}") from e


def _resolve_k(T: int, k: Union[int, float]) -> int:
    """
    Converts k (int or fraction) into an integer number of frames in [1, T].
    """
    if isinstance(k, float):
        if not (0.0 < k <= 1.0):
            raise ValueError(f"top-k fraction must be in (0,1]. Got {k}")
        kk = int(np.ceil(k * T))
    else:
        kk = int(k)
    return max(1, min(kk, T))


def aggregate_video_predictions(
    logits: np.ndarray,
    y_true: np.ndarray,
    video_ids: List[str],
    frame_ids: List[int],
    method: str = "mean_probs",
    smoothing: str = "none",
    smoothing_alpha: float = 0.7,
    *,
    # Optional: If provided, treat this class index as "Normal" and ignore it when selecting top-k frames.
    normal_class_idx: Optional[int] = None,
    # Controls how we score frames to pick the "top" ones:
    # - "max": max probability across all classes (your current behavior)
    # - "crime_max": max probability across non-normal classes (requires normal_class_idx)
    topk_score: str = "max",
) -> VideoAggResult:
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

        # Derive a single true label for the video: majority of frame labels
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
            # method can be:
            #   "topk_mean_probs" (default fraction=5%)
            #   "topk_mean_probs:20"
            #   "topk_mean_probs:0.05"
            k_raw = _parse_topk(method, default_k=0.05)
            T = probs_seq.shape[0]
            k = _resolve_k(T, k_raw)

            # score each frame
            if topk_score == "max":
                conf = np.max(probs_seq, axis=1)  # (T,)
            elif topk_score == "crime_max":
                if normal_class_idx is None:
                    raise ValueError("topk_score='crime_max' requires normal_class_idx.")
                C = probs_seq.shape[1]
                mask = np.ones((C,), dtype=bool)
                if 0 <= normal_class_idx < C:
                    mask[normal_class_idx] = False
                else:
                    raise ValueError(f"normal_class_idx out of range: {normal_class_idx} for C={C}")
                conf = np.max(probs_seq[:, mask], axis=1)  # (T,)
            else:
                raise ValueError(f"Unknown topk_score: {topk_score}")

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
