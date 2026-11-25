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


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (deg) at b between b->a and b->c, per frame; inputs (F, 2). 180 = straight."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    u, v = a - b, c - b
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = np.sum(u * v, axis=-1)
    return np.degrees(np.arctan2(np.abs(cross), dot))


def angular_velocity(angle_deg: np.ndarray, fps: float) -> np.ndarray:
    """Time derivative (deg/s) by central differences (one-sided at the ends)."""
    a = np.asarray(angle_deg, dtype=float)
    if a.shape[0] < 2:
        return np.zeros(a.shape[0])
    return np.gradient(a) * fps


def trunk_flexion(hip_c: np.ndarray, shoulder_c: np.ndarray, direction: int) -> np.ndarray:
    """Signed angle (deg) between the trunk (hip center -> shoulder center) and straight up
    (image y is down); positive when the shoulders are ahead of the hips in the throw direction."""
    d = np.asarray(shoulder_c, dtype=float) - np.asarray(hip_c, dtype=float)
    return np.degrees(np.arctan2(d[..., 0] * direction, -d[..., 1]))


def rotation_proxy(left_px: np.ndarray, right_px: np.ndarray) -> np.ndarray:
    """Rotation proxy (deg) from the left/right joint x distance w, normalized by its clip
    maximum: degrees(arccos(clip(w / max(w), 0, 1))). All-NaN input gives all NaN."""
    w = np.abs(np.asarray(left_px, dtype=float)[:, 0] - np.asarray(right_px, dtype=float)[:, 0])
    if not np.isfinite(w).any():
        return np.full(w.shape, np.nan)
    top = np.nanmax(w)
    if top <= 0:
        return np.full(w.shape, np.nan)
    return np.degrees(np.arccos(np.clip(w / top, 0.0, 1.0)))


def argmax_in(values: np.ndarray, lo: int | None, hi: int | None) -> int | None:
    """Index of the max of values within [lo, hi] (clipped to the array); None if the
    window is empty or has no finite value."""
    if lo is None or hi is None:
        return None
    lo, hi = max(int(lo), 0), min(int(hi), len(values) - 1)
    if hi < lo:
        return None
    seg = np.asarray(values[lo:hi + 1], dtype=float)
    if not np.isfinite(seg).any():
        return None
    return lo + int(np.nanargmax(seg))


def peak_frames(kp, fps: float, ppm: float, subject, events) -> dict[str, int | None]:
    """Frame of max linear speed of pelvis center, shoulder center, throwing elbow and
    wrist within [leg_lift_peak, release + 0.05 s]. All None if either event is None."""
    from .joints import idx, side_joints

    out: dict[str, int | None] = {k: None for k in ("pelvis_center", "shoulder_center", "elbow", "wrist")}
    if events.leg_lift_peak is None or events.release is None:
        return out
    xy = np.asarray(kp.xy, dtype=float)
    j = side_joints(subject.hand)
    tracks = {
        "pelvis_center": (xy[:, idx("left_hip")] + xy[:, idx("right_hip")]) / 2,
        "shoulder_center": (xy[:, idx("left_shoulder")] + xy[:, idx("right_shoulder")]) / 2,
        "elbow": xy[:, j["throw_elbow"]],
        "wrist": xy[:, j["throw_wrist"]],
    }
    hi = events.release + int(round(0.05 * fps))
    for key, pts in tracks.items():
        out[key] = argmax_in(speed(pts, fps, ppm), events.leg_lift_peak, hi)
    return out
