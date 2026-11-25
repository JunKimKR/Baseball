"""Energy flow per body segment, energy leaks and segment scores."""

from __future__ import annotations

import numpy as np

from .joints import idx, side_joints
from .kinematics import angular_velocity, peak_frames, rotation_proxy, speed  # noqa: F401
from .models import Leak, Metric, Norm, SegmentScore, Subject

SEGMENTS = ["legs", "pelvis", "trunk", "arm"]
SEGMENT_LABELS = {"legs": "다리", "pelvis": "골반", "trunk": "몸통", "arm": "팔"}

# segment -> norm whose mean normalizes that column
_COLUMN_NORM = {"legs": "pelvis_peak_speed_ms", "trunk": "shoulder_peak_speed_ms",
                "arm": "wrist_peak_speed_ms"}
# (peak key, segment, label) in the proximal-to-distal order
_ORDER = [("pelvis_center", "legs", "골반 중심"), ("shoulder_center", "trunk", "몸통"),
          ("elbow", "arm", "팔꿈치"), ("wrist", "arm", "손목")]
_SPEED_DROP = [("pelvis_peak_speed_ms", "legs", "pelvis_center", "골반 최고 속도"),
               ("shoulder_peak_speed_ms", "trunk", "shoulder_center", "몸통 최고 속도"),
               ("wrist_peak_speed_ms", "arm", "wrist", "손목 최고 속도")]


def energy_intensity(kp, fps: float, ppm: float, subject: Subject, direction: int,
                     norms: dict[str, Norm]) -> np.ndarray:
    """(F, 4) intensity in 0..1, columns = SEGMENTS."""
    xy = np.asarray(kp.xy, dtype=float)
    j = side_joints(subject.hand)
    lh, rh = xy[:, idx("left_hip")], xy[:, idx("right_hip")]
    raw = {
        "legs": speed((lh + rh) / 2, fps, ppm),
        "pelvis": np.abs(angular_velocity(rotation_proxy(lh, rh), fps)),
        "trunk": speed((xy[:, idx("left_shoulder")] + xy[:, idx("right_shoulder")]) / 2, fps, ppm),
        "arm": speed(xy[:, j["throw_wrist"]], fps, ppm),
    }
    cols = []
    for seg in SEGMENTS:
        col = np.nan_to_num(np.asarray(raw[seg], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        norm = norms.get(_COLUMN_NORM.get(seg, ""))
        denom = norm.mean if norm is not None and norm.mean > 0 else (col.max() if col.size else 0.0)
        col = col / denom if denom > 0 else np.zeros_like(col)
        cols.append(np.clip(col, 0.0, 1.0))
    return np.stack(cols, axis=1)


def detect_leaks(metrics: list[Metric], peaks: dict[str, int | None], fps: float) -> list[Leak]:
    leaks: list[Leak] = []
    if fps >= 60:
        for (ka, seg, name), (kb, _, nxt) in zip(_ORDER, _ORDER[1:]):
            a, b = peaks.get(ka), peaks.get(kb)
            if a is None or b is None or a <= b:
                continue
            ms = (a - b) * 1000.0 / fps
            leaks.append(Leak(seg, int(a), "out_of_order", min(1.0, ms / 30.0),
                              f"{name} 최고 속도가 {nxt}보다 {ms:.0f}ms 늦습니다"))
    by_name = {m.name: m for m in metrics}
    for metric, seg, peak_key, label in _SPEED_DROP:
        m, frame = by_name.get(metric), peaks.get(peak_key)
        if m is None or m.z is None or frame is None or not m.z < -1:
            continue
        leaks.append(Leak(seg, int(frame), "speed_drop", min(1.0, abs(m.z) / 3.0),
                          f"{label}가 기준보다 낮습니다 (z={m.z:.1f})"))
    return leaks


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def segment_scores(metrics: list[Metric]) -> list[SegmentScore]:
    out = []
    for seg in SEGMENTS:
        scored = [m for m in metrics if m.segment == seg and m.score is not None]
        if not scored:
            out.append(SegmentScore(seg, SEGMENT_LABELS[seg], None, "low"))
            continue
        score = round(sum(m.score for m in scored) / len(scored))
        conf = min((m.confidence for m in scored), key=_CONF_RANK.__getitem__)
        out.append(SegmentScore(seg, SEGMENT_LABELS[seg], score, conf))
    return out
