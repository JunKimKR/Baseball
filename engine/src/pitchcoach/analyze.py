"""End-to-end pipeline: video -> AnalysisResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import camera_motion_ratio
from .diagnosis import diagnose, load_rules
from .energy import detect_leaks, energy_intensity, segment_scores
from .events import build_phases, detect_events, throw_direction
from .joints import side_joints
from .kinematics import peak_frames
from .metrics import compute_metrics
from .models import AnalysisResult, AnalysisWarning, Energy, Subject
from .norms import load_norms, score_metrics
from .pose import MediaPipeEstimator, PoseEstimator
from .quality import check_quality
from .scale import pixels_per_meter
from .smoothing import smooth

CAMERA_MOTION_THRESHOLD = 0.05
CAMERA_MOTION_MESSAGE = "카메라가 움직였습니다. 속도 지표는 참고만 하세요. 삼각대로 고정해 촬영하세요."


def analyze(video_path: Path, subject: Subject,
            estimator: PoseEstimator | None = None) -> AnalysisResult:
    video_path = Path(video_path)
    estimator = estimator or MediaPipeEstimator()
    kp_raw, meta = estimator.estimate(video_path)

    key_joints = list(side_joints(subject.hand).values())
    warnings = check_quality(meta, kp_raw, subject, key_joints)
    if any(w.severity == "error" for w in warnings):
        return AnalysisResult(meta=meta, subject=subject, warnings=warnings, keypoints=kp_raw)

    kp = smooth(kp_raw, meta.fps)
    ppm = pixels_per_meter(kp, subject.height_cm)
    events, event_warnings = detect_events(kp, meta.fps, ppm, subject)
    warnings += event_warnings
    direction = throw_direction(kp, subject.hand, events)
    phases = build_phases(events, meta.n_frames, meta.fps)

    camera_moving = camera_motion_ratio(video_path, kp) > CAMERA_MOTION_THRESHOLD
    if camera_moving:
        warnings.append(AnalysisWarning("CAMERA_MOTION", "warn", CAMERA_MOTION_MESSAGE))

    norms = load_norms()
    metrics = score_metrics(
        compute_metrics(kp, events, meta.fps, ppm, subject, direction, camera_moving=camera_moving),
        norms, fps=meta.fps)
    peaks = peak_frames(kp, meta.fps, ppm, subject, events)
    energy = Energy(
        intensity=energy_intensity(kp, meta.fps, ppm, subject, direction, norms),
        leaks=detect_leaks(metrics, peaks, meta.fps),
        peaks=peaks,
    )
    return AnalysisResult(
        meta=meta, subject=subject, warnings=warnings, throw_direction=direction,
        keypoints=kp, events=events, phases=phases, metrics=metrics,
        segments=segment_scores(metrics), energy=energy,
        findings=diagnose(metrics, energy.leaks, load_rules()),
    )
