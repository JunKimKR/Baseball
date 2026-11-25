"""The fixed set of 15 pitching metrics."""

from __future__ import annotations

import math

import numpy as np

from .joints import idx, side_joints
from .kinematics import (angle, angular_velocity, argmax_in, peak_frames, rotation_proxy,
                         speed, trunk_flexion)
from .models import Events, Keypoints, Metric, Subject


def _d(name, label, unit, segment, confidence, timing=False):
    return {"name": name, "label": label, "unit": unit, "segment": segment,
            "confidence": confidence, "timing": timing}


METRIC_DEFS: list[dict] = [
    _d("stride_length_pct_height", "보폭(키 대비)", "%", "legs", "high"),
    _d("lead_knee_flexion_fp_deg", "착지 시 앞무릎 굴곡", "°", "legs", "high"),
    _d("lead_knee_extension_deg", "앞다리 블록(무릎 펴짐)", "°", "legs", "high"),
    _d("pelvis_peak_speed_ms", "골반 최고 속도", "m/s", "legs", "medium"),
    _d("pelvis_rotation_proxy_deg", "골반 회전(추정)", "°", "pelvis", "low"),
    _d("shoulder_rotation_proxy_deg", "어깨 회전(추정)", "°", "pelvis", "low"),
    _d("trunk_flexion_br_deg", "릴리스 시 몸통 숙임", "°", "trunk", "high"),
    _d("trunk_flexion_peak_vel_dps", "몸통 숙임 최고 각속도", "°/s", "trunk", "medium"),
    _d("shoulder_peak_speed_ms", "어깨 중심 최고 속도", "m/s", "trunk", "medium"),
    _d("elbow_ext_peak_vel_dps", "팔꿈치 펴짐 최고 각속도", "°/s", "arm", "medium"),
    _d("wrist_peak_speed_ms", "손목 최고 속도", "m/s", "arm", "medium"),
    _d("fp_to_br_ms", "착지→릴리스 시간", "ms", "arm", "high", True),
    _d("gap_pelvis_shoulder_ms", "골반→어깨 최고속도 간격", "ms", "trunk", "medium", True),
    _d("gap_shoulder_elbow_ms", "어깨→팔꿈치 최고속도 간격", "ms", "arm", "medium", True),
    _d("gap_elbow_wrist_ms", "팔꿈치→손목 최고속도 간격", "ms", "arm", "medium", True),
]


def _at(series: np.ndarray, frame: int | None):
    if frame is None or not 0 <= frame < len(series):
        return None
    return series[frame]


def _clean(value) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _gap_ms(a: int | None, b: int | None, fps: float):
    return None if a is None or b is None else (b - a) * 1000.0 / fps


def compute_metrics(kp: Keypoints, events: Events, fps: float, ppm: float, subject: Subject,
                    direction: int, camera_moving: bool = False) -> list[Metric]:
    xy = np.asarray(kp.xy, dtype=float)
    j = side_joints(subject.hand)
    height_m = subject.height_cm / 100.0
    fp, br, llp = events.foot_plant, events.release, events.leg_lift_peak
    after = lambda s: None if br is None else br + int(round(s * fps))  # noqa: E731

    hip_c = (xy[:, idx("left_hip")] + xy[:, idx("right_hip")]) / 2
    sh_c = (xy[:, idx("left_shoulder")] + xy[:, idx("right_shoulder")]) / 2
    knee = angle(xy[:, j["lead_hip"]], xy[:, j["lead_knee"]], xy[:, j["lead_ankle"]])
    elbow_vel = angular_velocity(
        angle(xy[:, j["throw_shoulder"]], xy[:, j["throw_elbow"]], xy[:, j["throw_wrist"]]), fps)
    trunk = trunk_flexion(hip_c, sh_c, direction)
    trunk_vel = angular_velocity(trunk, fps)
    pk = peak_frames(kp, fps, ppm, subject, events)

    def peak_val(series, lo, hi):
        i = argmax_in(series, lo, hi)
        return None if i is None else series[i]

    v: dict[str, float | None] = {}
    if fp is not None and 0 <= fp < len(xy):
        v["stride_length_pct_height"] = (
            (xy[fp, j["lead_ankle"], 0] - xy[fp, j["rear_ankle"], 0]) * direction
            / (height_m * ppm) * 100.0)
        v["lead_knee_flexion_fp_deg"] = 180.0 - knee[fp]
        v["pelvis_rotation_proxy_deg"] = _at(
            rotation_proxy(xy[:, idx("left_hip")], xy[:, idx("right_hip")]), fp)
        v["shoulder_rotation_proxy_deg"] = _at(
            rotation_proxy(xy[:, idx("left_shoulder")], xy[:, idx("right_shoulder")]), fp)
    if fp is not None and br is not None and 0 <= fp < len(xy) and 0 <= br < len(xy):
        v["lead_knee_extension_deg"] = knee[br] - knee[fp]
    v["trunk_flexion_br_deg"] = _at(trunk, br)
    v["pelvis_peak_speed_ms"] = (
        None if llp is None or br is None
        else peak_val(speed(hip_c, fps, ppm), llp, br))
    if fp is not None and br is not None:
        v["shoulder_peak_speed_ms"] = peak_val(speed(sh_c, fps, ppm), fp, br)
        v["trunk_flexion_peak_vel_dps"] = peak_val(trunk_vel, fp, after(0.1))
        v["elbow_ext_peak_vel_dps"] = peak_val(elbow_vel, fp, after(0.05))
        v["wrist_peak_speed_ms"] = peak_val(
            speed(xy[:, j["throw_wrist"]], fps, ppm), fp, after(0.05))
        v["fp_to_br_ms"] = _gap_ms(fp, br, fps)
    v["gap_pelvis_shoulder_ms"] = _gap_ms(pk["pelvis_center"], pk["shoulder_center"], fps)
    v["gap_shoulder_elbow_ms"] = _gap_ms(pk["shoulder_center"], pk["elbow"], fps)
    v["gap_elbow_wrist_ms"] = _gap_ms(pk["elbow"], pk["wrist"], fps)

    out = []
    for d in METRIC_DEFS:
        value, conf = _clean(v.get(d["name"])), d["confidence"]
        if d["timing"]:
            if fps < 60:
                value = None
            if fps < 120 and conf == "high":
                conf = "medium"
        if d["unit"] == "°/s" and fps < 60:
            conf = "low"
        if d["unit"] == "m/s" and (fps < 60 or camera_moving):
            conf = "low"
        out.append(Metric(d["name"], d["label"], value, d["unit"], conf, d["segment"]))
    return out
