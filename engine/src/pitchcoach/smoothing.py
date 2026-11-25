"""Keypoint smoothing: gap interpolation + zero-phase Butterworth low-pass."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .models import Keypoints

_ORDER = 4


def _interpolate(col: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Linear interpolation over ~ok samples; ends hold the nearest valid value."""
    t = np.arange(len(col))
    return np.interp(t, t[ok], col[ok])


def smooth(
    kp: Keypoints, fps: float, cutoff_hz: float = 15.0, vis_threshold: float = 0.5
) -> Keypoints:
    """Return smoothed coordinates; vis is returned unchanged.

    Frames with vis < vis_threshold are treated as missing and interpolated.
    A joint with no visible frame stays NaN. Clips too short for filtfilt are
    returned interpolated but unfiltered.
    """
    xy = np.array(kp.xy, dtype=float)
    vis = np.asarray(kp.vis)
    n_frames = xy.shape[0]
    cutoff = min(cutoff_hz, 0.4 * fps / 2)
    b, a = butter(_ORDER, cutoff / (fps / 2), btype="low")
    padlen = 3 * max(len(a), len(b))
    can_filter = n_frames > padlen

    out = np.full_like(xy, np.nan)
    for j in range(xy.shape[1]):
        ok = (vis[:, j] >= vis_threshold) & np.isfinite(xy[:, j]).all(axis=1)
        if not ok.any():
            continue
        for c in range(2):
            series = _interpolate(xy[:, j, c], ok)
            out[:, j, c] = filtfilt(b, a, series) if can_filter else series
    return Keypoints(xy=out, vis=kp.vis)


def longest_gap_s(kp: Keypoints, fps: float, joints: list[int], vis_threshold: float = 0.5) -> float:
    """Longest run (seconds) of consecutive frames where any single joint is missing."""
    vis = np.asarray(kp.vis)
    longest = 0
    for j in joints:
        run = 0
        for missing in vis[:, j] < vis_threshold:
            run = run + 1 if missing else 0
            longest = max(longest, run)
    return longest / fps
