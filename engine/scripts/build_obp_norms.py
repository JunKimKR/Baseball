"""Build the packaged 2D side-view norms (norms_side_v1.json) from OpenBiomechanics data.

Driveline's OpenBiomechanics Project (OBP) publishes 3D joint-center trajectories of fastball
pitches (360 Hz, meters; lab frame +X = 2nd base -> home, +Y = 1st-base side, +Z = up). Each
trial is projected onto the sagittal plane, as a side-view camera sees it, and then goes
through the same smoothing, event detection and metric code as a video, so that the norms
use the video pipeline's own definitions:

    3D -> (x, y) = (X, -Z) [m, ppm = 1] -> MediaPipe joint slots -> 240 Hz -> smooth()
       -> detect_events() -> throw_direction() -> compute_metrics() -> mean / sd per metric

OBP's own events (force-plate foot plant, ball release) are only compared with ours (printed).

Usage (from engine/):
    ../.venv/bin/python scripts/build_obp_norms.py \\
        --landmarks data/obp/raw/pitching_landmarks.zip \\
        --metadata data/obp/repo/baseball_pitching/data/metadata.csv \\
        --out src/pitchcoach/data/norms_side_v1.json

The data are licensed CC BY-NC-SA 4.0, and so is the generated file (engine/data/obp/README.md).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from pitchcoach.events import detect_events, throw_direction
from pitchcoach.joints import JOINTS, idx, side_joints
from pitchcoach.kinematics import speed
from pitchcoach.metrics import METRIC_DEFS, compute_metrics
from pitchcoach.models import Events, Keypoints, Subject
from pitchcoach.smoothing import smooth

REPO_URL = "https://github.com/drivelineresearch/openbiomechanics"
RELEASE_TAG = "dataset-v1"
LANDMARKS_URL = f"{REPO_URL}/releases/download/{RELEASE_TAG}/pitching_landmarks.zip"
LANDMARKS_SHA256 = "efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22"
LANDMARKS_MEMBER = "landmarks.csv"
LICENSE = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"

OBP_FPS = 360.0
NORM_FPS = 240.0               # metrics (timing included) are computed as for a 240 fps video
PPM = 1.0                      # coordinates are meters
FORWARD_AXIS, UP_AXIS = 0, 2   # OBP lab frame: +X 2nd base -> home, +Y 1st-base side, +Z up

# metadata.csv playing_level: "college or above" is in, below is out. Any other value stops
# the build, so that a new data release gets looked at.
INCLUDED_LEVELS = ("college", "independent", "milb")
EXCLUDED_LEVELS = ("high_school",)

# OBP landmark -> (body side, MediaPipe part). OBP names the arms throwing (no prefix) / glove
# and the legs rear / lead: the rear leg is on the throwing side, the lead leg on the glove
# side. The other OBP landmarks (hand_jc, glove_hand_jc, thorax_*, centerofmass) have no
# MediaPipe slot, and MediaPipe's face, hand and foot points get vis = 0.
OBP_JOINTS = {
    "shoulder_jc": ("throw", "shoulder"),
    "elbow_jc": ("throw", "elbow"),
    "wrist_jc": ("throw", "wrist"),
    "glove_shoulder_jc": ("glove", "shoulder"),
    "glove_elbow_jc": ("glove", "elbow"),
    "glove_wrist_jc": ("glove", "wrist"),
    "rear_hip": ("throw", "hip"),
    "rear_knee_jc": ("throw", "knee"),
    "rear_ankle_jc": ("throw", "ankle"),
    "lead_hip": ("glove", "hip"),
    "lead_knee_jc": ("glove", "knee"),
    "lead_ankle_jc": ("glove", "ankle"),
}
OPTIONAL_JOINTS = ("glove_elbow_jc", "glove_wrist_jc")

NORM_METRICS = [d["name"] for d in METRIC_DEFS if d["confidence"] != "low"]
HIGHER_KEYWORDS = ("speed", "_vel_", "stride", "knee_extension")
METHOD = ("OBP 3D 관절 중심(360 Hz, m)을 측면으로 투영(x = X: 2루→홈, y = −Z) → 240 Hz 선형 "
          "리샘플 → pitchcoach smooth(15 Hz 저역통과) → detect_events·throw_direction·"
          "compute_metrics(영상 분석과 같은 정의) → 지표별 평균·표본표준편차")


# --- per-trial pipeline ---

def project_sagittal(points_3d: np.ndarray, forward_axis: int, up_axis: int) -> np.ndarray:
    """(F, J, 3) lab coordinates -> (F, J, 2) side view: x = forward axis, y = -up axis (image
    y points down). The remaining (lateral) axis is dropped."""
    if forward_axis == up_axis:
        raise ValueError("forward_axis and up_axis must differ")
    p = np.asarray(points_3d, dtype=float)
    return np.stack([p[..., forward_axis], -p[..., up_axis]], axis=-1)


def mediapipe_joint(obp_name: str, hand: str) -> str | None:
    """MediaPipe joint name of an OBP landmark for a pitcher throwing with `hand`
    (right-hander: throwing arm and rear leg = right); None if it has no MediaPipe slot."""
    if hand not in ("R", "L"):
        raise ValueError(f"hand must be 'R' or 'L', got {hand!r}")
    if obp_name not in OBP_JOINTS:
        return None
    side, part = OBP_JOINTS[obp_name]
    throw_word, glove_word = ("right", "left") if hand == "R" else ("left", "right")
    return f"{throw_word if side == 'throw' else glove_word}_{part}"


def obp_keypoints(points_2d: np.ndarray, names: list[str], hand: str) -> Keypoints:
    """(F, J, 2) projected OBP landmarks -> Keypoints in MediaPipe order (meters). Joints
    without an OBP landmark are NaN with vis = 0."""
    missing = [n for n in OBP_JOINTS if n not in OPTIONAL_JOINTS and n not in names]
    if missing:
        raise ValueError(f"OBP landmarks missing: {', '.join(missing)}")
    pts = np.asarray(points_2d, dtype=float)
    xy = np.full((pts.shape[0], len(JOINTS), 2), np.nan)
    vis = np.zeros((pts.shape[0], len(JOINTS)))
    for k, name in enumerate(names):
        joint = mediapipe_joint(name, hand)
        if joint is not None:
            xy[:, idx(joint)] = pts[:, k]
            vis[:, idx(joint)] = np.isfinite(pts[:, k]).all(axis=1)
    return Keypoints(xy=xy, vis=vis)


def resample(kp: Keypoints, fps_in: float, fps_out: float) -> Keypoints:
    """Linear interpolation onto an fps_out grid that starts at the first frame and stops at
    or before the last one (no extrapolation). Joints with no finite value stay NaN, vis 0."""
    xy, vis = np.asarray(kp.xy, dtype=float), np.asarray(kp.vis, dtype=float)
    t_in = np.arange(xy.shape[0]) / fps_in
    n_out = int(np.floor((xy.shape[0] - 1) * fps_out / fps_in + 1e-9)) + 1
    t_out = np.arange(n_out) / fps_out
    out_xy = np.full((n_out,) + xy.shape[1:], np.nan)
    out_vis = np.zeros((n_out, xy.shape[1]))
    for j in range(xy.shape[1]):
        if not np.isfinite(xy[:, j]).any():
            continue
        out_vis[:, j] = np.interp(t_out, t_in, vis[:, j])
        for c in range(xy.shape[2]):
            out_xy[:, j, c] = np.interp(t_out, t_in, xy[:, j, c])
    return Keypoints(xy=out_xy, vis=out_vis)


def throw_wrist_is_fastest(kp: Keypoints, hand: str, frame: int, fps: float) -> bool:
    """Whether the throwing wrist is the fastest tracked joint at `frame`. At release nothing
    else we track moves faster, so this checks that OBP's throwing-arm landmarks were mapped
    to the throwing side."""
    xy, vis = np.asarray(kp.xy, dtype=float), np.asarray(kp.vis)
    tracked = np.flatnonzero((vis > 0).any(axis=0))
    speeds = np.array([speed(xy[:, j], fps, PPM)[frame] for j in tracked])
    if not np.isfinite(speeds).any():
        return False
    return int(tracked[np.nanargmax(speeds)]) == side_joints(hand)["throw_wrist"]


@dataclass
class TrialResult:
    """Metric values of one trial, or why it was skipped (values is then empty)."""
    values: dict[str, float | None]
    events: Events                 # frames at NORM_FPS, counted from the trial's first frame
    direction: int | None = None
    skip_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def process_trial(points_3d: np.ndarray, names: list[str], fps: float,
                  subject: Subject) -> TrialResult:
    """One OBP trial (F, J, 3 at `fps`) through the video pipeline at 240 Hz."""
    xy = project_sagittal(points_3d, FORWARD_AXIS, UP_AXIS)
    kp = smooth(resample(obp_keypoints(xy, names, subject.hand), fps, NORM_FPS), NORM_FPS)
    events, warnings = detect_events(kp, NORM_FPS, PPM, subject)
    codes = [w.code for w in warnings]
    if None in dataclasses.astuple(events):
        return TrialResult({}, events, None, "events_not_found", codes)
    direction = throw_direction(kp, subject.hand, events)
    if direction != 1:  # every OBP pitch goes toward +X
        return TrialResult({}, events, direction, "throw_direction_not_forward", codes)
    if not throw_wrist_is_fastest(kp, subject.hand, events.release, NORM_FPS):
        return TrialResult({}, events, direction, "throw_wrist_not_fastest", codes)
    metrics = compute_metrics(kp, events, NORM_FPS, PPM, subject, direction)
    return TrialResult({m.name: m.value for m in metrics}, events, direction, None, codes)


# --- aggregation ---

def norm_direction(name: str) -> str:
    """Speeds, angular velocities, stride and lead-knee extension: higher is better.
    Everything else is scored by its distance from the population mean ("target")."""
    return "higher" if any(k in name for k in HIGHER_KEYWORDS) else "target"


def in_population(level: str) -> bool:
    """College or above; an unknown playing level raises ValueError."""
    if level in INCLUDED_LEVELS:
        return True
    if level in EXCLUDED_LEVELS:
        return False
    raise ValueError(f"unknown playing level {level!r}: add it to INCLUDED_LEVELS or EXCLUDED_LEVELS")


@dataclass
class TrialInfo:
    session_pitch: str
    athlete: str       # metadata.csv user
    level: str         # metadata.csv playing_level
    height_cm: float   # metadata.csv session_height_m x 100
    hand: str          # poi_metrics.csv p_throws


def population_summary(infos: list[TrialInfo], skipped: dict[str, int]) -> dict:
    """Levels, trial and athlete counts of the trials the norms are built from."""
    return {
        "levels": sorted({i.level for i in infos}),
        "n_trials": len(infos),
        "n_athletes": len({i.athlete for i in infos}),
        "n_trials_skipped": sum(skipped.values()),
        "skipped": dict(sorted(skipped.items())),
    }


def _round(x: float) -> float:
    return float(f"{x:.5g}")


def build_norms(values: list[dict[str, float | None]], source: str, population: dict) -> dict:
    """The norms document (Task 5 format plus provenance): mean and sample sd over the trials
    of every metric whose default confidence is not "low"."""
    norms = {}
    for name in NORM_METRICS:
        v = np.array([x[name] for x in values if x.get(name) is not None], dtype=float)
        sd = _round(float(np.std(v, ddof=1))) if len(v) >= 2 else float("nan")
        if not sd > 0:
            raise ValueError(f"{name}: sd must be > 0 (n = {len(v)}, sd = {sd})")
        norms[name] = {"mean": _round(float(v.mean())), "sd": sd, "n": int(len(v)),
                       "direction": norm_direction(name), "source": source}
    return {"version": 1, "population": population, "source": source, "license": LICENSE,
            "license_url": LICENSE_URL, "method": METHOD, "norms": norms}


def write_norms(doc: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_string(commit: str) -> str:
    return (f"OpenBiomechanics Project (Driveline Baseball) 투구 데이터 — {REPO_URL}, "
            f"릴리스 {RELEASE_TAG} pitching_landmarks.zip, 저장소 커밋 {commit}, {LICENSE}")


# --- reading the OBP files ---

@dataclass
class ObpTrial:
    session_pitch: str
    names: list[str]          # landmark names, e.g. "lead_ankle_jc"
    points: np.ndarray        # (F, J, 3) meters, lab frame, OBP_FPS
    t0: float                 # time of the first frame (s)
    events: dict[str, float]  # OBP event times (s), e.g. "BR_time"; NaN when empty


def read_landmarks(zip_path: Path) -> Iterator[ObpTrial]:
    """Trials of landmarks.csv in pitching_landmarks.zip, in file order. A trial's rows must
    be contiguous and on a 360 Hz time grid."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(LANDMARKS_MEMBER) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        header = next(reader)
        col = {c: i for i, c in enumerate(header)}
        names = [c[:-2] for c in header
                 if c.endswith("_x") and f"{c[:-2]}_y" in col and f"{c[:-2]}_z" in col]
        layout = (col, names, [col[f"{n}_{a}"] for n in names for a in "xyz"],
                  [c for c in header if c.endswith("_time")])
        seen: set[str] = set()
        current, rows = None, []
        for row in reader:
            if row[col["session_pitch"]] != current:
                if rows:
                    yield _obp_trial(current, rows, *layout)
                current, rows = row[col["session_pitch"]], []
                if current in seen:
                    raise ValueError(f"rows of trial {current} are not contiguous")
                seen.add(current)
            rows.append(row)
        if rows:
            yield _obp_trial(current, rows, *layout)


def _obp_trial(sp, rows, col, names, coord_cols, event_cols) -> ObpTrial:
    t = np.array([r[col["time"]] for r in rows], dtype=float)
    if np.abs(t - t[0] - np.arange(len(t)) / OBP_FPS).max() > 1e-3:
        raise ValueError(f"{sp}: time column is not a {OBP_FPS:g} Hz grid")
    points = np.array([[r[i] for i in coord_cols] for r in rows], dtype=float)
    events = {c: float(rows[0][col[c]]) if rows[0][col[c]].strip() else float("nan")
              for c in event_cols}
    return ObpTrial(sp, names, points.reshape(len(rows), len(names), 3), float(t[0]), events)


def read_trial_info(metadata_csv: Path, poi_csv: Path) -> dict[str, TrialInfo]:
    """session_pitch -> athlete, playing level and height (metadata.csv) and throwing hand
    (p_throws in poi_metrics.csv; metadata.csv has no handedness column)."""
    with open(poi_csv, newline="", encoding="utf-8") as f:
        hands = {r["session_pitch"]: r["p_throws"] for r in csv.DictReader(f)}
    infos: dict[str, TrialInfo] = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp, height_m = r["session_pitch"], float(r["session_height_m"])
            hand = hands.get(sp)
            if sp in infos:
                raise ValueError(f"{sp}: duplicate row in {metadata_csv}")
            if not 1.4 <= height_m <= 2.3:
                raise ValueError(f"{sp}: session_height_m = {height_m} is not a height in meters")
            if hand not in ("R", "L"):
                raise ValueError(f"{sp}: p_throws = {hand!r} in {poi_csv}, expected R or L")
            infos[sp] = TrialInfo(sp, r["user"], r["playing_level"], height_m * 100.0, hand)
    return infos


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def obp_repo_commit(metadata_csv: Path) -> str:
    """HEAD commit of the OpenBiomechanics clone that holds metadata.csv."""
    def git(*args: str) -> str:
        cmd = ["git", "-C", str(Path(metadata_csv).resolve().parent), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    try:
        remote, commit = git("config", "--get", "remote.origin.url"), git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"cannot read the OBP repository commit ({e}); pass --obp-commit") from e
    if "drivelineresearch/openbiomechanics" not in remote:
        raise SystemExit(f"{metadata_csv} is not in an OpenBiomechanics clone (origin {remote!r}); "
                         "pass --obp-commit")
    return commit


# --- command line ---

def _describe(deltas_ms: list[float]) -> str:
    d = np.asarray(deltas_ms, dtype=float)
    d = d[np.isfinite(d)]
    if not d.size:
        return "no data"
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return (f"median {med:+.1f}, IQR [{q1:+.1f}, {q3:+.1f}], n = {d.size}, "
            f"|delta| > 40 ms: {int((np.abs(d) > 40).sum())}")


def _summary(doc: dict, excluded: Counter, warning_codes: Counter, fp_ms: list[float],
             br_ms: list[float]) -> str:
    pop = doc["population"]
    lines = [
        f"population {pop['levels']}: {pop['n_trials']} trials, {pop['n_athletes']} athletes; "
        f"skipped {pop['skipped']}; excluded levels {dict(excluded)}",
        f"event detector warnings: {dict(warning_codes)}",
        "event agreement, ours - OBP (ms):",
        f"  foot plant vs fp_100_time: {_describe(fp_ms)}",
        f"  release vs BR_time:        {_describe(br_ms)}",
        f"{'metric':30s} {'mean':>10s} {'sd':>9s} {'n':>5s}  direction",
    ]
    for name, e in doc["norms"].items():
        lines.append(f"{name:30s} {e['mean']:10.3f} {e['sd']:9.3f} {e['n']:5d}  {e['direction']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build norms_side_v1.json from OpenBiomechanics data.")
    p.add_argument("--landmarks", type=Path, required=True,
                   help=f"pitching_landmarks.zip from {LANDMARKS_URL}")
    p.add_argument("--metadata", type=Path, required=True,
                   help="baseball_pitching/data/metadata.csv of the OBP repository")
    p.add_argument("--poi", type=Path,
                   help="poi_metrics.csv with p_throws (default: <metadata dir>/poi/poi_metrics.csv)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-sha256", default=LANDMARKS_SHA256)
    p.add_argument("--obp-commit", help="OBP repository commit (default: HEAD of the clone)")
    args = p.parse_args(argv)

    digest = sha256_of(args.landmarks)
    if digest != args.expected_sha256:
        raise SystemExit(f"{args.landmarks}: sha256 {digest} != expected {args.expected_sha256}")
    commit = args.obp_commit or obp_repo_commit(args.metadata)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"not a full git commit hash: {commit!r}")
    infos = read_trial_info(args.metadata, args.poi or args.metadata.parent / "poi" / "poi_metrics.csv")

    used, values = [], []
    skipped, excluded, warning_codes = Counter(), Counter(), Counter()
    fp_ms, br_ms = [], []
    for trial in read_landmarks(args.landmarks):
        if trial.session_pitch not in infos:
            raise ValueError(f"{trial.session_pitch}: no row in {args.metadata}")
        info = infos[trial.session_pitch]
        if not in_population(info.level):
            excluded[info.level] += 1
            continue
        res = process_trial(trial.points, trial.names, OBP_FPS,
                            Subject(info.height_cm, info.hand, "arm"))
        warning_codes.update(res.warnings)
        if res.skip_reason:
            skipped[res.skip_reason] += 1
            continue
        used.append(info)
        values.append(res.values)
        fp_ms.append(1000 * (trial.t0 + res.events.foot_plant / NORM_FPS - trial.events["fp_100_time"]))
        br_ms.append(1000 * (trial.t0 + res.events.release / NORM_FPS - trial.events["BR_time"]))

    doc = build_norms(values, source_string(commit), population_summary(used, skipped))
    write_norms(doc, args.out)
    print(_summary(doc, excluded, warning_codes, fp_ms, br_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
