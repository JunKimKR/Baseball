"""Pitch events and phases.

Detection is anchored on the release and works backwards (release -> foot plant ->
leg-lift peak -> MER), so walking or set-up shuffles before and after the pitch are
never mistaken for the stride or for another pitch.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .joints import side_joints
from .kinematics import speed
from .metrics import MIN_TIMING_FPS
from .models import AnalysisWarning, Events, Keypoints, Phase, Subject

MIN_RELEASE_SPEED_MS = 3.0   # no release unless the wrist reaches this speed
PEAK_FRACTION = 0.5          # release (and another pitch) = wrist peak >= 0.5 x max
FOOT_PLANT_WINDOW_S = 0.4    # foot plant is searched in [release - 0.4 s, release]
PLANT_SPEED_MS = 0.5         # lead ankle below this is "planted"
PLANT_FRAMES = 3             # ... for at least this many consecutive frames
LEG_LIFT_WINDOW_S = 2.0      # leg-lift peak is searched in [foot plant - 2 s, foot plant]
NEXT_PITCH_AFTER_S = 1.0     # a wrist peak later than this after release = another pitch
FOLLOW_THROUGH_S = 0.5

EVENT_LABELS = {
    "leg_lift_peak": "레그 리프트 최고점",
    "foot_plant": "앞발 착지",
    "mer_est": "최대 외회전(추정)",
    "release": "릴리스",
}


def detect_events(
    kp: Keypoints, fps: float, ppm: float, subject: Subject
) -> tuple[Events, list[AnalysisWarning]]:
    """Events of the first pitch in the clip (frame indices or None) and warnings."""
    sj = side_joints(subject.hand)
    xy = np.asarray(kp.xy, dtype=float)
    warnings: list[AnalysisWarning] = []
    leg_lift_peak = foot_plant = mer_est = None

    release, another_pitch = _release(speed(xy[:, sj["throw_wrist"]], fps, ppm), fps)
    # below 60 fps smoothing flattens the release peak, so walking after the pitch can pass the
    # 0.5 x max test: the warning would be a guess there
    if another_pitch and fps >= MIN_TIMING_FPS:
        warnings.append(AnalysisWarning(
            "MULTIPLE_PITCHES", "warn", "두 번째 투구로 보이는 동작이 있어 첫 투구만 분석했습니다."))
    if release is not None:
        foot_plant = _foot_plant(speed(xy[:, sj["lead_ankle"]], fps, ppm), release, fps)
    if foot_plant is not None:
        leg_lift_peak = _leg_lift_peak(xy[:, sj["lead_knee"], 1], foot_plant, fps)
        direction = throw_direction(kp, subject.hand, Events(leg_lift_peak, foot_plant, None, release))
        behind = (xy[:, sj["throw_wrist"], 0] - xy[:, sj["throw_shoulder"], 0]) * direction
        mer_est = _first_argmin(behind, foot_plant, release)

    events = Events(leg_lift_peak, foot_plant, mer_est, release)
    for name, label in EVENT_LABELS.items():
        if getattr(events, name) is None:
            warnings.append(AnalysisWarning(
                "EVENT_NOT_FOUND", "warn",
                f"{label} 시점을 찾지 못했습니다. 투구 동작 전체가 화면에 담기도록 촬영하세요."))
    return events, warnings


def throw_direction(kp: Keypoints, hand: str, events: Events) -> int:
    """+1 if the pitcher throws toward frame right (+x), -1 toward the left.

    Sign of the lead ankle's x shift from leg-lift peak to foot plant; without those
    events, sign of the throwing wrist's x velocity at release; otherwise +1.
    """
    sj = side_joints(hand)
    xy = np.asarray(kp.xy, dtype=float)
    if events.leg_lift_peak is not None and events.foot_plant is not None:
        ankle_x = xy[:, sj["lead_ankle"], 0]
        sign = _sign(ankle_x[events.foot_plant] - ankle_x[events.leg_lift_peak])
        if sign:
            return sign
    if events.release is not None and xy.shape[0] >= 2:
        sign = _sign(np.gradient(xy[:, sj["throw_wrist"], 0])[events.release])
        if sign:
            return sign
    return 1


def build_phases(events: Events, n_frames: int, fps: float) -> list[Phase]:
    """Contiguous phases between events; a phase is left out if an end event is None."""
    llp, fp, mer, br = events.leg_lift_peak, events.foot_plant, events.mer_est, events.release
    end = None if br is None else min(br + int(round(FOLLOW_THROUGH_S * fps)), n_frames - 1)
    spans = [
        ("leg_lift", "레그 리프트", 0, llp),
        ("stride", "스트라이드", llp, fp),
        ("arm_cocking", "팔 코킹", fp, mer),
        ("acceleration", "가속·릴리스", mer, br),
        ("follow_through", "팔로스루", br, end),
    ]
    return [Phase(name, label, int(start), int(stop))
            for name, label, start, stop in spans if start is not None and stop is not None]


def _release(wrist_speed: np.ndarray, fps: float) -> tuple[int | None, bool]:
    """First wrist-speed local max >= PEAK_FRACTION x max, and whether another such
    peak comes more than NEXT_PITCH_AFTER_S later."""
    s = np.where(np.isfinite(wrist_speed), wrist_speed, 0.0)
    top = float(s.max()) if s.size else 0.0
    if top < MIN_RELEASE_SPEED_MS:
        return None, False
    peaks, _ = find_peaks(s, height=PEAK_FRACTION * top)
    if peaks.size == 0:
        return None, False
    release = int(peaks[0])
    return release, bool((peaks > release + NEXT_PITCH_AFTER_S * fps).any())


def _foot_plant(ankle_speed: np.ndarray, release: int, fps: float) -> int | None:
    """In [release - 0.4 s, release]: after the lead ankle has moved (>= 0.5 m/s), the first
    frame starting a run of >= 3 frames below 0.5 m/s."""
    lo = max(0, release - int(round(FOOT_PLANT_WINDOW_S * fps)))
    window = ankle_speed[lo:release + 1]
    moving = np.flatnonzero(window >= PLANT_SPEED_MS)
    if moving.size == 0:
        return None
    still = window < PLANT_SPEED_MS
    for i in range(moving[0] + 1, len(window) - PLANT_FRAMES + 1):
        if still[i:i + PLANT_FRAMES].all():
            return lo + i
    return None


def _leg_lift_peak(knee_y: np.ndarray, foot_plant: int, fps: float) -> int | None:
    """Frame where the lead knee is highest (smallest image y) in [FP - 2 s, FP]."""
    lo = max(0, foot_plant - int(round(LEG_LIFT_WINDOW_S * fps)))
    return _first_argmin(knee_y, lo, foot_plant)


def _first_argmin(values: np.ndarray, lo: int, hi: int) -> int | None:
    """Frame of the minimum of values[lo..hi] (inclusive), ignoring NaN; None if all NaN."""
    segment = values[lo:hi + 1]
    if not np.isfinite(segment).any():
        return None
    return lo + int(np.nanargmin(segment))


def _sign(value: float) -> int:
    return int(np.sign(value)) if np.isfinite(value) else 0
