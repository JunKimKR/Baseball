"""Synthetic side-view pitch for tests: a 33-joint stick figure driven by smooth time functions.

Shared by the tests of Tasks 3, 4, 6, 7 and 9, so this contract is exact:

* All 33 joints have vis = 1. Positions are pixels (``ppm`` per meter, y down) and every
  trajectory has continuous velocity over the whole clip, between pitches included.
* Each pitch starts with STAND_S seconds of still standing whose ankle-midpoint -> nose
  distance is 0.89 x height (matches ``scale.pixels_per_meter``).
* The lead knee is highest, above the lead hip, at ``leg_lift_peak_s``. The lead ankle
  strides toward the target (peak ~4 m/s) and comes to rest at ``foot_plant_s``.
* Pelvis center, shoulder center, throwing elbow and throwing wrist each have one
  Gaussian speed peak at foot plant + ``peak_offsets_ms``. The wrist peak (~22 m/s) is
  the release and the clip's top wrist speed. At mer_est = release - 30 ms the wrist is
  furthest behind the throwing shoulder along the throw.
* The returned Events are the first pitch's truth frames, round(time_s * fps).

Built in meters for a right-hander throwing toward +x (y down, standing ankle midpoint at
the origin), then mirrored for direction=-1 and side-swapped for hand="L".
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from pitchcoach.joints import JOINTS
from pitchcoach.models import Events, Keypoints
from pitchcoach.scale import NOSE_ANKLE_RATIO

# --- timing (seconds) ---
STAND_S = 0.2                     # still standing at the start of each pitch
MIN_LLP_TO_FP_S = 0.2             # room for the foot to come down and the arm to swing
DEFAULT_PEAK_OFFSETS_MS = {"pelvis": 10, "shoulder": 60, "elbow": 125, "wrist": 150}
MER_BEFORE_RELEASE_S = 0.030      # truth mer_est = release - 30 ms
LIFT_END_BEFORE_FP_S = 0.10       # lead foot is down again (still sliding) by FP - 0.1 s
STRIDE_START = 0.25               # stride starts this fraction of LLP->FP after LLP
FOOT_STOP_S = 0.03                # lead ankle decelerates to rest over the last 30 ms
ARM_SWING_AFTER_LLP_S = 0.10      # hands break and the arms swing from LLP + 0.1 s ...
ARM_SWING_BEFORE_FP_S = 0.05      # ... to FP - 0.05 s
LAYBACK_RISE_S = 0.05             # wrist lays back from FP - 50 ms ...
LAYBACK_FALL_S = 0.025            # ... until 25 ms after mer_est (before release)
ROTATION_SIGMA_S = 0.06           # hips / shoulders open around their speed peaks
NEXT_PITCH_AFTER_RELEASE_S = 1.5  # n_pitches > 1: next pitch starts 1.5 s after release,
RECOVER_AFTER_RELEASE_S = 0.6     # after gliding back to the standing pose from here
WALK_AFTER_RELEASE_S = 0.6
WALK_S = 1.5
WALK_STEPS = 3                    # rear, lead, rear foot
WALK_ANKLE_PEAK_MS = 1.4
TAIL_S = 0.6                      # clip ends 0.6 s after the last action

# --- Gaussian speed profiles: (peak m/s, sigma s, direction in degrees below +x) ---
PELVIS = (2.5, 0.08, 5.0)
SHOULDER = (4.5, 0.07, 10.0)
ELBOW = (11.0, 0.04, 5.0)
WRIST = (20.0, 0.015, 18.0)       # relative to the throwing shoulder

# --- body, in fractions of height (x toward the target, y down) ---
STANCE_HALF = 0.08
PELVIS_Y, HIP_HALF = -0.50, 0.09
SHOULDER_Y, SHOULDER_HALF = -0.78, 0.11
ROTATION_CLOSURE = 0.8            # hip / shoulder x-width shrinks by 80 % as they open
STRIDE = 0.60
LIFT_ANKLE, LIFT_KNEE = (0.0, -0.18), (0.10, -0.25)
KNEE_BEND = (0.03, 0.0)           # knee = hip-ankle midpoint + bend (+ lift)
WALK_FOOT_LIFT = 0.03
THROW_ELBOW, THROW_ELBOW_SWING = (-0.08, -0.62), (-0.10, -0.06)
THROW_WRIST_REL, THROW_WRIST_SWING = (0.15, 0.12), (-0.15, -0.22)  # from throwing shoulder
GLOVE_ELBOW_REL, GLOVE_ELBOW_REACH, GLOVE_ELBOW_TUCK = (0.03, 0.16), (0.12, -0.06), (-0.14, 0.08)
GLOVE_WRIST_REL, GLOVE_WRIST_REACH, GLOVE_WRIST_TUCK = (-0.06, 0.12), (0.24, -0.06), (-0.26, 0.10)
NOSE = (0.0, -NOSE_ANKLE_RATIO - SHOULDER_Y)  # from the shoulder center
FACE = {"eye_inner": (-0.01, -0.125), "eye": (-0.02, -0.126), "eye_outer": (-0.03, -0.125),
        "ear": (-0.06, -0.115), "mouth": (-0.01, -0.085)}
HAND = {"pinky": (0.03, 0.02), "index": (0.04, 0.0), "thumb": (0.025, -0.015)}
FOOT = {"heel": (-0.03, 0.02), "foot_index": (0.08, 0.025)}

# --- placement in the image (meters) ---
X0_M, FRAME_W_M, GROUND_Y_M = 1.0, 4.0, 2.1


def make_pitch(fps=240, hand="R", direction=1, ppm=500.0, height_cm=185.0,
               leg_lift_peak_s=0.5, foot_plant_s=1.25, peak_offsets_ms=None,
               n_pitches=1, walk_after=False) -> tuple[Keypoints, Events]:
    """Synthetic keypoints and the first pitch's truth events.

    ``peak_offsets_ms`` may give only some of pelvis/shoulder/elbow/wrist; the rest keep
    their defaults. ``n_pitches`` > 1 repeats the pitch 1.5 s after each release;
    ``walk_after`` walks forward for 1.5 s from 0.6 s after the last release.
    """
    offsets = {k: v / 1000 for k, v in {**DEFAULT_PEAK_OFFSETS_MS, **(peak_offsets_ms or {})}.items()}
    llp, fp = leg_lift_peak_s, foot_plant_s
    br = fp + offsets["wrist"]
    if hand not in ("R", "L") or direction not in (1, -1) or n_pitches < 1:
        raise ValueError("hand must be R/L, direction +1/-1, n_pitches >= 1")
    if not STAND_S < llp < fp - MIN_LLP_TO_FP_S or offsets["wrist"] < MER_BEFORE_RELEASE_S:
        raise ValueError("need 0.2 s < leg_lift_peak_s < foot_plant_s - 0.2 s and wrist offset >= 30 ms")

    h = height_cm / 100.0
    period = br + NEXT_PITCH_AFTER_RELEASE_S
    last_release = (n_pitches - 1) * period + br
    end = last_release + (WALK_AFTER_RELEASE_S + WALK_S if walk_after else 0.0) + TAIL_S
    t = np.arange(int(round(end * fps)) + 1) / fps

    q = _standing(h, len(t))
    for k in range(n_pitches):
        tau = t - k * period
        motion = _pitch_motion(tau, h, llp, fp, offsets)
        if k < n_pitches - 1:  # glide back to the standing pose before the next pitch
            settled = _pitch_motion(np.array([br + RECOVER_AFTER_RELEASE_S]), h, llp, fp, offsets)
            back = _ease(tau, br + RECOVER_AFTER_RELEASE_S, period)[:, None]
            motion = {key: m - settled[key] * back for key, m in motion.items()}
        for key, m in motion.items():
            q[key] += m
    if walk_after:
        _add_walk(q, t, last_release + WALK_AFTER_RELEASE_S, h)

    xy = _to_pixels(_joints(q, h), hand, direction, ppm)

    def frame(time_s: float) -> int:
        return int(round(time_s * fps))

    truth = Events(frame(llp), frame(fp), frame(br - MER_BEFORE_RELEASE_S), frame(br))
    return Keypoints(xy=xy, vis=np.ones(xy.shape[:2])), truth


# --- smooth time primitives ---

def _ease(t, t0, t1):
    """Quintic smoothstep 0 -> 1 on [t0, t1]; zero velocity and acceleration at both ends."""
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def _bump(t, t0, tp, t1):
    """0 -> 1 -> 0 on [t0, t1] (sin^2 halves): one rounded peak at tp, zero velocity at the ends."""
    rise = np.sin(0.5 * np.pi * np.clip((t - t0) / (tp - t0), 0.0, 1.0)) ** 2
    fall = np.cos(0.5 * np.pi * np.clip((t - tp) / (t1 - tp), 0.0, 1.0)) ** 2
    return np.where(t < tp, rise, fall)


def _gauss_step(t, center, sigma):
    """0 -> 1 with a Gaussian velocity profile centered at ``center``."""
    return ndtr((t - center) / sigma)


def _pulse_travel(t, a, b, c, d):
    """Distance covered at unit peak speed when speed rises (sin^2) on [a, b], holds 1 on
    [b, c] and falls (cos^2) on [c, d]; velocity and acceleration are continuous."""
    rise, hold, fall = (np.clip(t, lo, hi) - lo for lo, hi in ((a, b), (b, c), (c, d)))
    out = hold + rise / 2 + fall / 2
    if b > a:
        out -= (b - a) / (2 * np.pi) * np.sin(np.pi * rise / (b - a))
    if d > c:
        out += (d - c) / (2 * np.pi) * np.sin(np.pi * fall / (d - c))
    return out


def _travel(tau, profile, center):
    """(F, 2) displacement along a fixed direction with a Gaussian speed profile."""
    peak, sigma, angle = profile
    unit = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    return peak * sigma * np.sqrt(2 * np.pi) * _gauss_step(tau, center, sigma)[:, None] * unit


# --- body model (canonical: right-hander throwing toward +x, meters, y down) ---

def _standing(h, n):
    pose = {
        "pelvis": (0.0, PELVIS_Y * h), "hip_half": (HIP_HALF * h, 0.0),
        "shoulder": (0.0, SHOULDER_Y * h), "shoulder_half": (SHOULDER_HALF * h, 0.0),
        "lead_ankle": (STANCE_HALF * h, 0.0), "rear_ankle": (-STANCE_HALF * h, 0.0),
        "knee_lift": (0.0, 0.0),
        "throw_elbow": np.multiply(THROW_ELBOW, h), "throw_wrist_rel": np.multiply(THROW_WRIST_REL, h),
        "glove_elbow_rel": np.multiply(GLOVE_ELBOW_REL, h), "glove_wrist_rel": np.multiply(GLOVE_WRIST_REL, h),
    }
    return {key: np.tile(np.asarray(p, dtype=float), (n, 1)) for key, p in pose.items()}


def _pitch_motion(tau, h, llp, fp, offsets):
    """Displacement from the standing pose for one pitch, local time ``tau``."""
    t_pelvis, t_shoulder, t_elbow, br = (fp + offsets[k] for k in ("pelvis", "shoulder", "elbow", "wrist"))
    mer = br - MER_BEFORE_RELEASE_S
    lift = _bump(tau, STAND_S, llp, fp - LIFT_END_BEFORE_FP_S)[:, None]
    swing = _ease(tau, llp + ARM_SWING_AFTER_LLP_S, fp - ARM_SWING_BEFORE_FP_S)[:, None]
    tuck = _ease(tau, fp, br)[:, None]
    forward = np.array([1.0, 0.0])

    # Lead ankle: speed rises slowly, then stops within FOOT_STOP_S, at rest at fp.
    s0 = llp + STRIDE_START * (fp - llp)
    stride = STRIDE * h * _pulse_travel(tau, s0, fp - FOOT_STOP_S, fp - FOOT_STOP_S, fp) / ((fp - s0) / 2)

    # Wrist layback relative to the throwing shoulder: its backward x speed equals the
    # forward Gaussian's speed at mer, so wrist-minus-shoulder x falls until mer, then rises.
    peak_x = WRIST[0] * np.cos(np.radians(WRIST[2]))
    layback = peak_x * np.exp(-0.5 * (MER_BEFORE_RELEASE_S / WRIST[1]) ** 2) \
        * _pulse_travel(tau, fp - LAYBACK_RISE_S, fp, mer, mer + LAYBACK_FALL_S)

    def opening(center):
        return -ROTATION_CLOSURE * _gauss_step(tau, center, ROTATION_SIGMA_S)[:, None]

    def body(vec):
        return np.multiply(vec, h)

    return {
        "pelvis": _travel(tau, PELVIS, t_pelvis),
        "hip_half": opening(t_pelvis) * body((HIP_HALF, 0.0)),
        "shoulder": _travel(tau, SHOULDER, t_shoulder),
        "shoulder_half": opening(t_shoulder) * body((SHOULDER_HALF, 0.0)),
        "lead_ankle": lift * body(LIFT_ANKLE) + stride[:, None] * forward,
        "rear_ankle": np.zeros((len(tau), 2)),
        "knee_lift": lift * body(LIFT_KNEE),
        "throw_elbow": swing * body(THROW_ELBOW_SWING) + _travel(tau, ELBOW, t_elbow),
        "throw_wrist_rel": swing * body(THROW_WRIST_SWING) - layback[:, None] * forward
        + _travel(tau, WRIST, br),
        "glove_elbow_rel": swing * body(GLOVE_ELBOW_REACH) + tuck * body(GLOVE_ELBOW_TUCK),
        "glove_wrist_rel": swing * body(GLOVE_WRIST_REACH) + tuck * body(GLOVE_WRIST_TUCK),
    }


def _add_walk(q, t, start, h):
    """Alternating steps (rear, lead, ...) at WALK_ANKLE_PEAK_MS; the body follows the feet."""
    step = WALK_S / WALK_STEPS
    body = np.zeros(len(t))
    for i in range(WALK_STEPS):
        a = start + i * step
        mid, b = a + step / 2, a + step
        travel = WALK_ANKLE_PEAK_MS * _pulse_travel(t, a, mid, mid, b)
        foot = q["rear_ankle" if i % 2 == 0 else "lead_ankle"]
        foot[:, 0] += travel
        foot[:, 1] -= WALK_FOOT_LIFT * h * _bump(t, a, mid, b)
        body += travel / 2
    for key in ("pelvis", "shoulder", "throw_elbow"):
        q[key][:, 0] += body


def _joints(q, h):
    """All 33 joints by role: T_ = throwing side, G_ = glove (lead) side."""
    j = {"nose": q["shoulder"] + np.multiply(NOSE, h)}
    j["G_hip"], j["T_hip"] = q["pelvis"] + q["hip_half"], q["pelvis"] - q["hip_half"]
    j["G_shoulder"], j["T_shoulder"] = q["shoulder"] + q["shoulder_half"], q["shoulder"] - q["shoulder_half"]
    j["G_ankle"], j["T_ankle"] = q["lead_ankle"], q["rear_ankle"]
    bend = np.multiply(KNEE_BEND, h)
    j["G_knee"] = (j["G_hip"] + j["G_ankle"]) / 2 + bend + q["knee_lift"]
    j["T_knee"] = (j["T_hip"] + j["T_ankle"]) / 2 + bend
    j["T_elbow"] = q["throw_elbow"]
    j["T_wrist"] = j["T_shoulder"] + q["throw_wrist_rel"]
    j["G_elbow"] = j["G_shoulder"] + q["glove_elbow_rel"]
    j["G_wrist"] = j["G_shoulder"] + q["glove_wrist_rel"]
    for side, shift in (("T", -0.005), ("G", 0.005)):
        for part, (dx, dy) in FACE.items():
            j[f"{side}_{part}"] = q["shoulder"] + np.multiply((dx + shift, dy), h)
        for part, off in HAND.items():
            j[f"{side}_{part}"] = j[f"{side}_wrist"] + np.multiply(off, h)
        for part, off in FOOT.items():
            j[f"{side}_{part}"] = j[f"{side}_ankle"] + np.multiply(off, h)
    return j


def _role(name, throw_word):
    """MediaPipe joint name -> role key ("left_wrist" -> "T_wrist" for a left-hander)."""
    for word in ("left", "right"):
        side = "T" if word == throw_word else "G"
        if name.startswith(word + "_"):
            return f"{side}_{name[len(word) + 1:]}"
        if name.endswith("_" + word):  # mouth_left / mouth_right
            return f"{side}_{name[:-len(word) - 1]}"
    return name


def _to_pixels(j, hand, direction, ppm):
    throw_word = "right" if hand == "R" else "left"
    xy = np.stack([j[_role(name, throw_word)] for name in JOINTS], axis=1)  # (F, 33, 2)
    x = X0_M + xy[..., 0] if direction == 1 else FRAME_W_M - X0_M - xy[..., 0]
    return np.stack([x, GROUND_Y_M + xy[..., 1]], axis=-1) * ppm
