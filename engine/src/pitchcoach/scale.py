"""Pixel-to-meter scale from the pitcher's known height."""

from __future__ import annotations

import numpy as np

from .joints import idx
from .models import Keypoints

# Nose-to-ankle-midpoint distance as a fraction of standing height.
NOSE_ANKLE_RATIO = 0.89


def pixels_per_meter(kp: Keypoints, height_cm: float, n_frames: int = 15) -> float:
    """Median nose-to-ankle-midpoint pixel distance over the first n_frames,
    divided by 0.89 * height in meters."""
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")
    xy = kp.xy[:n_frames]
    ankles = (xy[:, idx("left_ankle")] + xy[:, idx("right_ankle")]) / 2
    dist = np.linalg.norm(xy[:, idx("nose")] - ankles, axis=1)
    return float(np.nanmedian(dist) / (NOSE_ANKLE_RATIO * height_cm / 100.0))
