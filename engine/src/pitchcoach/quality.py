"""Input quality checks producing Korean, actionable warnings."""

from __future__ import annotations

import numpy as np

from .metrics import FULL_RATE_FPS, MIN_TIMING_FPS
from .models import AnalysisWarning, Keypoints, Subject, VideoMeta
from .smoothing import longest_gap_s

VIS_THRESHOLD = 0.5
MIN_DURATION_S = 1.0
MAX_GAP_S = 0.1
MIN_PERSON_FRACTION = 0.8


def check_quality(
    meta: VideoMeta, kp_raw: Keypoints, subject: Subject, key_joints: list[int]
) -> list[AnalysisWarning]:
    """Warnings about the input. Frame-rate rules and the duration use the analysis fps (the
    capture fps of a slow-motion export), not the container's playback fps."""
    warnings: list[AnalysisWarning] = []
    fps = meta.analysis_fps

    if fps < FULL_RATE_FPS:
        warnings.append(AnalysisWarning(
            "LOW_FPS", "warn",
            "프레임 레이트가 120fps 미만이라 정확도가 떨어집니다. 가능하면 120fps 이상 슬로모션으로 촬영하세요."))
    if fps < MIN_TIMING_FPS:
        warnings.append(AnalysisWarning(
            "TIMING_DISABLED", "warn",
            "프레임 레이트가 60fps 미만이라 타이밍 지표를 표시하지 않고, 속도·각도 지표는 참고값으로만 "
            "보여 줍니다. 60fps 이상으로 촬영하세요."))
    if subject.camera_side != "arm":
        warnings.append(AnalysisWarning(
            "GLOVE_SIDE_CAMERA", "warn",
            "글러브 쪽에서 촬영된 영상입니다. 투구하는 팔 쪽 옆에서 촬영하면 더 정확합니다."))
    if longest_gap_s(kp_raw, fps, key_joints, VIS_THRESHOLD) > MAX_GAP_S:
        warnings.append(AnalysisWarning(
            "KEYPOINTS_MISSING", "warn",
            "주요 관절이 0.1초 넘게 인식되지 않은 구간이 있습니다. 몸 전체가 화면에 나오고 가려지지 않게 촬영하세요."))

    vis = np.asarray(kp_raw.vis)
    if len(key_joints) and vis.shape[0]:
        detected = vis[:, key_joints].mean(axis=1) >= VIS_THRESHOLD
        no_person = detected.mean() < MIN_PERSON_FRACTION
    else:
        no_person = True
    if no_person:
        warnings.append(AnalysisWarning(
            "NO_PERSON", "error",
            "투수를 인식하지 못했습니다. 투수 전신이 화면에 크게 나오도록 다시 촬영하세요."))
    if meta.n_frames / fps < MIN_DURATION_S:
        warnings.append(AnalysisWarning(
            "TOO_SHORT", "error",
            "영상이 1초보다 짧습니다. 투구 전후 동작이 모두 담기도록 길게 촬영하세요."))
    return warnings
