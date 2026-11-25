"""Pixel-to-meter scale from the pitcher's known height."""

from __future__ import annotations

import numpy as np

from .joints import idx
from .models import Keypoints

# Nose-to-ankle-midpoint distance as a fraction of standing height.
NOSE_ANKLE_RATIO = 0.89
SCALE_FRAMES = 15             # the scale is taken from the first frames (standing upright)
SCALE_MIN_RATIO = 0.9         # start distance below 90 % of the clip's 95th percentile: unsure


def _nose_ankle_px(xy: np.ndarray) -> np.ndarray:
    ankles = (xy[:, idx("left_ankle")] + xy[:, idx("right_ankle")]) / 2
    return np.linalg.norm(xy[:, idx("nose")] - ankles, axis=1)


def pixels_per_meter(kp: Keypoints, height_cm: float, n_frames: int = SCALE_FRAMES) -> float:
    """Median nose-to-ankle-midpoint pixel distance over the first n_frames,
    divided by 0.89 * height in meters."""
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")
    dist = _nose_ankle_px(np.asarray(kp.xy[:n_frames], dtype=float))
    return float(np.nanmedian(dist) / (NOSE_ANKLE_RATIO * height_cm / 100.0))


def is_scale_uncertain(kp: Keypoints, n_frames: int = SCALE_FRAMES,
                       min_ratio: float = SCALE_MIN_RATIO) -> bool:
    """Whether the height-based scale is doubtful: the start distance (median over the first
    n_frames, as pixels_per_meter uses it) is below min_ratio x the clip's 95th percentile of the
    nose-to-ankle-midpoint distance. Then the pitcher was not upright at the start, or the
    camera distance or zoom changed. Also True when the distance cannot be measured."""
    dist = _nose_ankle_px(np.asarray(kp.xy, dtype=float))
    start = dist[:n_frames]
    if not (np.isfinite(start).any() and np.isfinite(dist).any()):
        return True
    return not np.nanmedian(start) >= min_ratio * np.nanpercentile(dist, 95)
