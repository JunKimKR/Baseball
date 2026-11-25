"""Basic kinematic time series from pixel keypoints."""

from __future__ import annotations

import numpy as np


def speed(points_px: np.ndarray, fps: float, ppm: float) -> np.ndarray:
    """Speed (m/s) of one point track (F, 2) in pixels, by central differences
    (one-sided at the clip ends)."""
    pts = np.asarray(points_px, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0])
    velocity = np.gradient(pts, axis=0) * (fps / ppm)
    return np.linalg.norm(velocity, axis=1)
