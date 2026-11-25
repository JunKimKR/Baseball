"""Data models and the result JSON (schema v1) serialization."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .joints import JOINTS

SCHEMA_VERSION = 1


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int
    norm_fps: float | None = None  # frame rate of the norms the metrics were scored against

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


@dataclass
class Subject:
    height_cm: float
    hand: Literal["R", "L"]
    camera_side: Literal["arm", "glove"]


@dataclass
class Keypoints:
    xy: np.ndarray   # (F, 33, 2) pixel coordinates
    vis: np.ndarray  # (F, 33)


@dataclass
class AnalysisWarning:
    code: str
    severity: Literal["info", "warn", "error"]
    message: str


@dataclass
class Events:
    leg_lift_peak: int | None = None
    foot_plant: int | None = None
    mer_est: int | None = None
    release: int | None = None


@dataclass
class Phase:
    name: str
    label: str
    start: int
    end: int


@dataclass
class Norm:
    mean: float
    sd: float
    direction: Literal["higher", "lower", "target"]
    source: str


@dataclass
class Metric:
    name: str
    label: str
    value: float | None
    unit: str
    confidence: Literal["high", "medium", "low"]
    segment: str
    score: int | None = None
    z: float | None = None
    norm: Norm | None = None


@dataclass
class SegmentScore:
    segment: str
    label: str
    score: int | None
    confidence: Literal["high", "medium", "low"]


@dataclass
class Leak:
    segment: str
    frame: int
    kind: Literal["out_of_order", "speed_drop"]
    severity: float
    message: str


def _empty_peaks() -> dict[str, int | None]:
    return {"pelvis_center": None, "shoulder_center": None, "elbow": None, "wrist": None}


@dataclass
class Energy:
    segments: list[str] = field(default_factory=lambda: ["legs", "pelvis", "trunk", "arm"])
    intensity: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # (F, 4), 0..1
    leaks: list[Leak] = field(default_factory=list)
    peaks: dict[str, int | None] = field(default_factory=_empty_peaks)


@dataclass
class SelfTest:
    id: str
    instruction: str
    options: list[dict[str, str]]  # [{"id": ..., "label": ...}, ...]
    implies_cause_on: str
    source: str


@dataclass
class Exercise:
    id: str
    name: str
    sets_reps: str
    description: str
    source: str


@dataclass
class Cause:
    id: str
    title: str
    self_test: SelfTest
    exercises: list[Exercise] = field(default_factory=list)


@dataclass
class Finding:
    issue_id: str
    title: str
    evidence: str
    causes: list[Cause] = field(default_factory=list)


@dataclass
class AnalysisResult:
    meta: VideoMeta
    subject: Subject
    warnings: list[AnalysisWarning] = field(default_factory=list)
    throw_direction: int = 1
    keypoints: Keypoints | None = None
    events: Events = field(default_factory=Events)
    phases: list[Phase] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    segments: list[SegmentScore] = field(default_factory=list)
    energy: Energy = field(default_factory=Energy)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Result JSON per schema v1: numpy -> lists, NaN/inf -> None."""
        if self.keypoints is None:
            frames: list = []
        else:
            xy = np.asarray(self.keypoints.xy, dtype=float)
            vis = np.asarray(self.keypoints.vis, dtype=float)
            frames = np.concatenate([xy, vis[..., None]], axis=-1)
        meta = _plain(self.meta)
        meta["duration_s"] = self.meta.duration_s
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": meta,
            "subject": _plain(self.subject),
            "warnings": _plain(self.warnings),
            "throw_direction": self.throw_direction,
            "keypoints": {"joints": list(JOINTS), "frames": _plain(frames)},
            "events": _plain(self.events),
            "phases": _plain(self.phases),
            "metrics": _plain(self.metrics),
            "segments": _plain(self.segments),
            "energy": _plain(self.energy),
            "findings": _plain(self.findings),
        }


def _plain(obj: Any) -> Any:
    """Recursively convert to JSON-safe builtins (numpy -> list, NaN/inf -> None)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
