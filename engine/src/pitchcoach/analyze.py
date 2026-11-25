"""End-to-end pipeline: video -> AnalysisResult."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .camera import camera_motion_ratio
from .diagnosis import diagnose, load_rules
from .energy import detect_leaks, energy_intensity, segment_scores
from .events import build_phases, detect_events, throw_direction
from .joints import side_joints
from .kinematics import peak_frames
from .metrics import compute_metrics
from .models import AnalysisResult, AnalysisWarning, Energy, Events, Keypoints, Metric, Subject
from .norms import load_norms_for_fps, score_metrics
from .pose import MediaPipeEstimator, PoseEstimator
from .quality import check_quality
from .scale import is_scale_uncertain, pixels_per_meter
from .smoothing import smooth

SCALE_MESSAGE = ("영상 처음의 자세가 이후 동작보다 작게 잡혀 키로 환산한 크기가 부정확할 수 있습니다"
                 "(처음에 숙이고 있었거나 카메라 거리·확대가 바뀐 경우). 보폭과 속도(m/s) 지표는 참고만 하세요. "
                 "녹화를 시작한 뒤 0.5초 이상 똑바로 선 자세로 시작하세요.")
CAMERA_MOTION_THRESHOLD = 0.05
CAMERA_MOTION_MESSAGE = ("카메라가 움직였습니다. 속도 지표와 릴리스 시 몸통 숙임은 참고만 하세요. "
                         "삼각대로 고정해 촬영하세요.")


def score_pitch(kp: Keypoints, events: Events, fps: float, ppm: float, subject: Subject,
                direction: int, camera_moving: bool = False,
                scale_uncertain: bool = False) -> tuple[list[Metric], Energy, float]:
    """Metrics of one detected pitch, scored against the packaged norms built for `fps`, and its
    energy flow. Returns (metrics, energy, norm_fps)."""
    norms, norm_fps = load_norms_for_fps(fps)
    metrics = score_metrics(
        compute_metrics(kp, events, fps, ppm, subject, direction, camera_moving=camera_moving,
                        scale_uncertain=scale_uncertain),
        norms, fps=fps, norm_fps=norm_fps)
    peaks = peak_frames(kp, fps, ppm, subject, events)
    energy = Energy(
        intensity=energy_intensity(kp, fps, ppm, subject, direction, norms),
        leaks=detect_leaks(metrics, peaks, fps),
        peaks=peaks,
    )
    return metrics, energy, norm_fps


def analyze(video_path: Path, subject: Subject, estimator: PoseEstimator | None = None,
            capture_fps: float | None = None) -> AnalysisResult:
    """Analyze one side-view pitching video.

    `capture_fps`: the frame rate the clip was shot at, for slow-motion clips whose file says
    otherwise (e.g. 240 fps exported as a 30 fps file). Every time-based computation uses it;
    the container fps (meta.fps) only maps frames to playback time. None = the container fps.
    """
    video_path = Path(video_path)
    estimator = estimator or MediaPipeEstimator()
    kp_raw, meta = estimator.estimate(video_path)
    if capture_fps is not None:
        meta = dataclasses.replace(meta, capture_fps=float(capture_fps))
    fps = meta.analysis_fps

    key_joints = list(side_joints(subject.hand).values())
    warnings = check_quality(meta, kp_raw, subject, key_joints)
    if any(w.severity == "error" for w in warnings):
        return AnalysisResult(meta=meta, subject=subject, warnings=warnings, keypoints=kp_raw)

    kp = smooth(kp_raw, fps)
    ppm = pixels_per_meter(kp, subject.height_cm)
    scale_uncertain = is_scale_uncertain(kp)
    if scale_uncertain:
        warnings.append(AnalysisWarning("SCALE_UNCERTAIN", "warn", SCALE_MESSAGE))
    events, event_warnings = detect_events(kp, fps, ppm, subject)
    warnings += event_warnings
    direction = throw_direction(kp, subject.hand, events)
    phases = build_phases(events, meta.n_frames, fps)

    camera_moving = camera_motion_ratio(video_path, kp) > CAMERA_MOTION_THRESHOLD
    if camera_moving:
        warnings.append(AnalysisWarning("CAMERA_MOTION", "warn", CAMERA_MOTION_MESSAGE))

    metrics, energy, norm_fps = score_pitch(kp, events, fps, ppm, subject, direction,
                                            camera_moving=camera_moving,
                                            scale_uncertain=scale_uncertain)
    return AnalysisResult(
        meta=dataclasses.replace(meta, norm_fps=norm_fps), subject=subject, warnings=warnings,
        throw_direction=direction,
        keypoints=kp, events=events, phases=phases, metrics=metrics,
        segments=segment_scores(metrics), energy=energy,
        findings=diagnose(metrics, energy.leaks, load_rules()),
    )
