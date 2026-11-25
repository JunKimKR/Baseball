"""Tests for scripts/build_obp_norms.py: OBP 3D landmarks -> 2D sagittal norms."""

import csv
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pitchcoach.events import detect_events
from pitchcoach.joints import idx, side_joints
from pitchcoach.metrics import METRIC_DEFS, compute_metrics
from pitchcoach.models import Keypoints, Subject
from pitchcoach.norms import load_norms
from pitchcoach.smoothing import smooth
from synth import make_pitch

ENGINE = Path(__file__).resolve().parents[1]
PACKAGED_FPS = (30.0, 60.0, 120.0, 240.0)
PACKAGED = {fps: ENGINE / "src" / "pitchcoach" / "data" / f"norms_side_v1_{fps:g}hz.json"
            for fps in PACKAGED_FPS}
NORM_NAMES = [d["name"] for d in METRIC_DEFS if d["confidence"] != "low"]
TIMING = {d["name"] for d in METRIC_DEFS if d["timing"]}
TIMING_NAMES = [n for n in NORM_NAMES if n in TIMING]
HIGHER = {"stride_length_pct_height", "lead_knee_extension_deg", "pelvis_peak_speed_ms",
          "trunk_flexion_peak_vel_dps", "shoulder_peak_speed_ms", "wrist_peak_speed_ms"}
SUBJ_R = Subject(185.0, "R", "arm")


def _load_script():
    path = ENGINE / "scripts" / "build_obp_norms.py"
    spec = importlib.util.spec_from_file_location("build_obp_norms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bon = _load_script()

# OBP landmark -> (side, MediaPipe part); T = throwing-arm side, G = glove side. Written out
# here independently of the script, so the tests check the script's mapping.
OBP_ROLES = {
    "shoulder_jc": ("T", "shoulder"), "elbow_jc": ("T", "elbow"), "wrist_jc": ("T", "wrist"),
    "glove_shoulder_jc": ("G", "shoulder"), "glove_elbow_jc": ("G", "elbow"),
    "glove_wrist_jc": ("G", "wrist"),
    "lead_hip": ("G", "hip"), "lead_knee_jc": ("G", "knee"), "lead_ankle_jc": ("G", "ankle"),
    "rear_hip": ("T", "hip"), "rear_knee_jc": ("T", "knee"), "rear_ankle_jc": ("T", "ankle"),
}
UNMAPPED = ["hand_jc", "thorax_ap", "centerofmass"]  # in the OBP file, not used for 2D
OFFSETS_A = {"pelvis": 0, "shoulder": 45, "elbow": 110, "wrist": 140}
OFFSETS_C = {"pelvis": 20, "shoulder": 80, "elbow": 135, "wrist": 165}


def fake_obp_trial(hand="R", peak_offsets_ms=None, x_scale=1.0, seed=0):
    """A make_pitch pitch at 360 Hz as OBP-style 3D landmarks: X = x * x_scale (forward),
    Z = -y (up), Y = random lateral values. Returns (names, points (F, J, 3), truth events)."""
    kp, truth = make_pitch(fps=360, ppm=1.0, hand=hand, peak_offsets_ms=peak_offsets_ms)
    side = {"T": "right", "G": "left"} if hand == "R" else {"T": "left", "G": "right"}
    rng = np.random.default_rng(seed)
    xy = [kp.xy[:, idx(f"{side[s]}_{part}")] for s, part in OBP_ROLES.values()]
    xy += [rng.uniform(0.0, 2.0, size=(kp.xy.shape[0], 2)) for _ in UNMAPPED]
    xy = np.stack(xy, axis=1)
    lateral = rng.uniform(-0.5, 0.5, size=xy.shape[:2])
    points = np.stack([xy[..., 0] * x_scale, lateral, -xy[..., 1]], axis=-1)
    return list(OBP_ROLES) + UNMAPPED, points, truth


def test_project_sagittal_drops_lateral():
    f = 6
    pts = np.zeros((f, 2, 3))
    pts[:, 0] = [0.3, 0.0, 1.2]
    pts[:, 1] = [-0.4, 0.0, 0.1]
    pts[:, :, 1] = np.linspace(-1.0, 1.0, f)[:, None]  # moves along the lateral axis only
    out = bon.project_sagittal(pts, forward_axis=0, up_axis=2)
    assert out.shape == (f, 2, 2)
    np.testing.assert_allclose(out[:, 0], np.tile([0.3, -1.2], (f, 1)))  # y = -up
    np.testing.assert_allclose(out[:, 1], np.tile([-0.4, -0.1], (f, 1)))
    np.testing.assert_allclose(bon.project_sagittal(pts, 1, 0)[:, 0, 1], -0.3)


@pytest.mark.parametrize("hand", ["R", "L"])
def test_mediapipe_mapping_matches_side_joints(hand):
    sj = side_joints(hand)
    for obp, role in [("shoulder_jc", "throw_shoulder"), ("elbow_jc", "throw_elbow"),
                      ("wrist_jc", "throw_wrist"), ("lead_hip", "lead_hip"),
                      ("lead_knee_jc", "lead_knee"), ("lead_ankle_jc", "lead_ankle"),
                      ("rear_hip", "rear_hip"), ("rear_knee_jc", "rear_knee"),
                      ("rear_ankle_jc", "rear_ankle")]:
        assert idx(bon.mediapipe_joint(obp, hand)) == sj[role], obp
    glove = "left" if hand == "R" else "right"
    for part in ("shoulder", "elbow", "wrist"):
        assert bon.mediapipe_joint(f"glove_{part}_jc", hand) == f"{glove}_{part}"
    for name in UNMAPPED:
        assert bon.mediapipe_joint(name, hand) is None


def test_obp_keypoints_fills_mapped_joints_only():
    names, points, _ = fake_obp_trial(hand="L")
    kp = bon.obp_keypoints(bon.project_sagittal(points, 0, 2), names, "L")
    assert kp.xy.shape == (points.shape[0], 33, 2)
    mapped = {idx(bon.mediapipe_joint(n, "L")) for n in OBP_ROLES}
    assert len(mapped) == 12
    for j in range(33):
        assert (kp.vis[:, j] == (1.0 if j in mapped else 0.0)).all()
        assert np.isfinite(kp.xy[:, j]).all() == (j in mapped)
    np.testing.assert_allclose(kp.xy[:, idx("left_wrist"), 1], -points[:, names.index("wrist_jc"), 2])
    with pytest.raises(ValueError, match="lead_ankle_jc"):
        bon.obp_keypoints(np.zeros((3, 1, 2)), ["wrist_jc"], "R")


def test_resample_360_to_240_linear():
    n = 361  # 1 s at 360 Hz
    t = np.arange(n) / 360.0
    xy = np.full((n, 33, 2), np.nan)
    vis = np.zeros((n, 33))
    xy[:, 0, 0], xy[:, 0, 1], vis[:, 0] = 2.0 * t, -t, 1.0
    out = bon.resample(Keypoints(xy, vis), 360.0, 240.0)
    t2 = np.arange(241) / 240.0
    assert out.xy.shape == (241, 33, 2) and out.vis.shape == (241, 33)
    np.testing.assert_allclose(out.xy[:, 0, 0], 2.0 * t2, atol=1e-12)
    np.testing.assert_allclose(out.xy[:, 0, 1], -t2, atol=1e-12)
    assert (out.vis[:, 0] == 1.0).all() and (out.vis[:, 1:] == 0.0).all()
    assert np.isnan(out.xy[:, 1:]).all()


@pytest.mark.parametrize("norm_fps", [240.0, 120.0, 60.0])
def test_process_trial_matches_direct_video_pipeline(norm_fps):
    """OBP path (360 Hz 3D -> project -> map -> norm_fps -> smooth) == the video path at norm_fps."""
    names, points, _ = fake_obp_trial()
    res = bon.process_trial(points, names, 360.0, SUBJ_R, norm_fps=norm_fps)
    assert res.skip_reason is None and res.direction == 1

    kp, _ = make_pitch(fps=norm_fps, ppm=1.0)
    kp = smooth(kp, norm_fps)
    events, _ = detect_events(kp, norm_fps, 1.0, SUBJ_R)
    direct = {m.name: m.value for m in compute_metrics(kp, events, norm_fps, 1.0, SUBJ_R, 1)}
    assert res.events == events
    assert sorted(res.values) == sorted(direct)
    for name in NORM_NAMES:
        a, b = res.values[name], direct[name]
        tol = 1e-6 if name in TIMING else max(0.005 * abs(b), 0.05)  # only interpolation differs
        assert abs(a - b) <= tol, (name, a, b)


def test_process_trial_left_hander_same_values():
    names, points, _ = fake_obp_trial(hand="R")
    names_l, points_l, _ = fake_obp_trial(hand="L")
    r = bon.process_trial(points, names, 360.0, SUBJ_R)
    left = bon.process_trial(points_l, names_l, 360.0, Subject(185.0, "L", "arm"))
    assert left.skip_reason is None and left.direction == 1
    for name in NORM_NAMES:
        assert left.values[name] == pytest.approx(r.values[name]), name


def test_process_trial_skip_reasons():
    names, points, _ = fake_obp_trial()
    still = np.repeat(points[:1], points.shape[0], axis=0)
    res = bon.process_trial(still, names, 360.0, SUBJ_R)
    assert res.skip_reason == "events_not_found" and res.values == {}
    backwards = points * np.array([-1.0, 1.0, 1.0])  # throws toward -X: wrong forward axis
    res = bon.process_trial(backwards, names, 360.0, SUBJ_R)
    assert res.skip_reason == "throw_direction_not_forward" and res.values == {}


def test_throw_wrist_is_fastest():
    n = 5
    xy, vis = np.zeros((n, 33, 2)), np.zeros((n, 33))
    for name, v in (("right_wrist", 20.0), ("left_wrist", 5.0), ("right_elbow", 10.0)):
        xy[:, idx(name), 0] = v * np.arange(n) / 240.0
        vis[:, idx(name)] = 1.0
    kp = Keypoints(xy, vis)
    assert bon.throw_wrist_is_fastest(kp, "R", 2, 240.0)
    assert not bon.throw_wrist_is_fastest(kp, "L", 2, 240.0)


def test_norm_directions_follow_brief():
    for name in NORM_NAMES:
        assert bon.norm_direction(name) == ("higher" if name in HIGHER else "target"), name


def test_population_levels():
    assert all(bon.in_population(lv) for lv in ("college", "independent", "milb"))
    assert not bon.in_population("high_school")
    with pytest.raises(ValueError, match="little_league"):
        bon.in_population("little_league")


def _info(sp, athlete, level, hand="R", height_cm=185.0):
    return bon.TrialInfo(session_pitch=sp, athlete=athlete, level=level, height_cm=height_cm,
                         hand=hand)


def test_build_norms_from_fake_trials(tmp_path):
    values = []
    variants = [(OFFSETS_A, 0.95), (None, 1.0), (OFFSETS_C, 1.05)]
    for seed, (offsets, scale) in enumerate(variants):
        names, points, _ = fake_obp_trial(peak_offsets_ms=offsets, x_scale=scale, seed=seed)
        res = bon.process_trial(points, names, 360.0, SUBJ_R)
        assert res.skip_reason is None
        values.append(res.values)
    infos = [_info("1_1", "1", "college"), _info("1_2", "1", "college"), _info("2_1", "2", "milb")]
    population = bon.population_summary(infos, {"events_not_found": 1})
    doc = bon.build_norms(values, "OBP test source", population, 240.0)
    path = tmp_path / "norms.json"
    bon.write_norms(doc, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw["norms"]) == NORM_NAMES
    for name, e in raw["norms"].items():
        assert e["sd"] > 0, name
        assert e["source"] == "OBP test source" and e["n"] == 3
        assert e["direction"] == ("higher" if name in HIGHER else "target")
    assert raw["version"] == 1 and raw["source"] == "OBP test source"
    assert raw["norm_fps"] == 240.0 and raw["not_computed"] == []
    assert "240 Hz" in raw["method"] and "15 Hz" in raw["method"]
    assert raw["license"] == "CC BY-NC-SA 4.0"
    assert raw["population"] == {"levels": ["college", "milb"], "n_trials": 3, "n_athletes": 2,
                                 "n_trials_skipped": 1, "skipped": {"events_not_found": 1}}
    loaded = load_norms(path)
    assert list(loaded) == NORM_NAMES
    stride = [v["stride_length_pct_height"] for v in values]
    assert loaded["stride_length_pct_height"].mean == pytest.approx(np.mean(stride), abs=1e-3)
    assert loaded["stride_length_pct_height"].sd == pytest.approx(np.std(stride, ddof=1), abs=1e-3)


def test_build_norms_rejects_degenerate_metric():
    names, points, _ = fake_obp_trial()
    res = bon.process_trial(points, names, 360.0, SUBJ_R)
    with pytest.raises(ValueError, match="sd"):
        bon.build_norms([res.values, res.values], "s", {}, 240.0)


def _three_trial_values():
    out = []
    for seed, (offsets, scale) in enumerate([(OFFSETS_A, 0.95), (None, 1.0), (OFFSETS_C, 1.05)]):
        names, points, _ = fake_obp_trial(peak_offsets_ms=offsets, x_scale=scale, seed=seed)
        out.append(bon.process_trial(points, names, 360.0, SUBJ_R).values)
    return out


def test_build_norms_below_60fps_omits_timing_metrics():
    """Below 60 fps compute_metrics leaves timing metrics None, so the norms have none."""
    values = [{k: (None if k in TIMING else v) for k, v in x.items()} for x in _three_trial_values()]
    doc = bon.build_norms(values, "s", {}, 30.0)
    assert doc["norm_fps"] == 30.0 and doc["not_computed"] == TIMING_NAMES
    assert list(doc["norms"]) == [n for n in NORM_NAMES if n not in TIMING]
    assert "30 Hz" in doc["method"] and "6 Hz" in doc["method"]  # smooth caps the cutoff at 0.2 fps
    with pytest.raises(ValueError, match="fp_to_br_ms"):  # a missing metric at >= 60 fps is an error
        bon.build_norms(values, "s", {}, 60.0)


# --- end to end: main() on a small OBP-format dataset ---

EVENT_COLUMNS = ["pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"]


def _write_obp_dataset(root: Path, trials):
    """trials: (session_pitch, user, level, hand, height_m, offsets, x_scale, has_fp_events)."""
    header, rows, meta, poi = None, [], [], []
    for k, (sp, user, level, hand, height_m, offsets, scale, has_fp) in enumerate(trials):
        names, points, truth = fake_obp_trial(hand, offsets, scale, seed=k)
        if header is None:
            header = (["session_pitch", "time"] + [f"{n}_{a}" for n in names for a in "xyz"]
                      + EVENT_COLUMNS)
        s = lambda frame: f"{frame / 360.0:.4f}"  # noqa: E731
        ev = [s(truth.leg_lift_peak), s(truth.foot_plant - 7) if has_fp else "",
              s(truth.foot_plant) if has_fp else "", s(truth.mer_est), s(truth.release),
              s(truth.release + 14)]
        for i in range(points.shape[0]):
            rows.append([sp, f"{i / 360.0:.4f}"] + [f"{v:.5f}" for v in points[i].ravel()] + ev)
        meta.append([user, user, sp, "90.0", str(height_m), "21.0", level, "88.0", "x.c3d", "m.c3d"])
        poi.append([sp, user, hand, "FF", "88.0"])
    buf = [",".join(header)] + [",".join(r) for r in rows]
    zip_path = root / "raw" / "pitching_landmarks.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("landmarks.csv", "\r\n".join(buf) + "\r\n")
    data = root / "repo" / "baseball_pitching" / "data"
    (data / "poi").mkdir(parents=True)
    with open(data / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "session", "session_pitch", "session_mass_kg", "session_height_m",
                    "age_yrs", "playing_level", "pitch_speed_mph", "filename_new", "modelname_new"])
        w.writerows(meta)
    with open(data / "poi" / "poi_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_pitch", "session", "p_throws", "pitch_type", "pitch_speed_mph"])
        w.writerows(poi)
    return zip_path, data / "metadata.csv"


@pytest.fixture
def obp_dataset(tmp_path):
    return _write_obp_dataset(tmp_path, [
        ("11_1", "11", "college", "R", 1.85, OFFSETS_A, 0.95, True),
        ("11_2", "11", "college", "R", 1.85, None, 1.0, False),
        ("12_1", "12", "milb", "L", 1.90, OFFSETS_C, 1.05, True),
        ("13_1", "13", "high_school", "R", 1.80, None, 1.0, True),
    ])


def _main_args(zip_path, meta_path, out, sha, *extra):
    return ["--landmarks", str(zip_path), "--metadata", str(meta_path), "--out", str(out),
            "--expected-sha256", sha, "--obp-commit", "c" * 40, *extra]


def test_main_end_to_end(obp_dataset, tmp_path, capsys):
    zip_path, meta_path = obp_dataset
    out = tmp_path / "out" / "norms.json"
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert bon.main(_main_args(zip_path, meta_path, out, sha)) == 0

    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["population"] == {"levels": ["college", "milb"], "n_trials": 3, "n_athletes": 2,
                                 "n_trials_skipped": 0, "skipped": {}}
    assert list(raw["norms"]) == NORM_NAMES
    assert "c" * 40 in raw["source"] and "dataset-v1" in raw["source"]
    assert all(e["source"] == raw["source"] and e["n"] == 3 for e in raw["norms"].values())
    # make_pitch strides ~0.76 x its 185 cm height; metadata heights are meters -> cm
    assert 70 < raw["norms"]["stride_length_pct_height"]["mean"] < 80
    assert raw["norm_fps"] == 240.0
    printed = capsys.readouterr().out
    assert "foot plant" in printed and "release" in printed


def test_main_fps_option(obp_dataset, tmp_path):
    zip_path, meta_path = obp_dataset
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    out = tmp_path / "norms60.json"
    assert bon.main(_main_args(zip_path, meta_path, out, sha, "--fps", "60")) == 0
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["norm_fps"] == 60.0 and "60 Hz" in raw["method"]
    assert list(raw["norms"]) == NORM_NAMES
    with pytest.raises(SystemExit):
        bon.main(_main_args(zip_path, meta_path, tmp_path / "x.json", sha, "--fps", "500"))


def test_main_rejects_wrong_sha256(obp_dataset, tmp_path):
    zip_path, meta_path = obp_dataset
    out = tmp_path / "norms.json"
    with pytest.raises(SystemExit, match="sha256"):
        bon.main(_main_args(zip_path, meta_path, out, "0" * 64))
    assert not out.exists()


# --- the packaged, OBP-derived norms (built by the script from the real data) ---

def test_packaged_norm_files_are_exactly_the_four_rates():
    data = ENGINE / "src" / "pitchcoach" / "data"
    assert sorted(p.name for p in data.glob("norms*.json")) == sorted(p.name for p in PACKAGED.values())


@pytest.mark.parametrize("fps", PACKAGED_FPS)
def test_packaged_norms_provenance(fps):
    raw = json.loads(PACKAGED[fps].read_text(encoding="utf-8"))
    assert raw["norm_fps"] == fps
    expected = NORM_NAMES if fps >= 60 else [n for n in NORM_NAMES if n not in TIMING]
    assert list(raw["norms"]) == expected
    assert raw["not_computed"] == [n for n in NORM_NAMES if n not in expected]
    assert f"{fps:g} Hz" in raw["method"]
    assert raw["license"] == "CC BY-NC-SA 4.0"
    pop = raw["population"]
    assert pop["levels"] == ["college", "independent", "milb"]
    assert pop["n_trials"] >= 300 and pop["n_athletes"] >= 50
    src = raw["source"]
    assert "github.com/drivelineresearch/openbiomechanics" in src and "dataset-v1" in src
    assert re.search(r"\b[0-9a-f]{40}\b", src)
    for name, e in raw["norms"].items():
        assert e["source"] == src and e["sd"] > 0 and e["n"] >= 300, name
        assert e["direction"] == ("higher" if name in HIGHER else "target"), name


@pytest.mark.parametrize("fps", PACKAGED_FPS)
def test_packaged_norms_physically_plausible(fps):
    """Guards against axis/unit/mapping regressions when the norms are rebuilt."""
    m = {k: e["mean"] for k, e in json.loads(PACKAGED[fps].read_text(encoding="utf-8"))["norms"].items()}
    assert 60 < m["stride_length_pct_height"] < 95
    assert 20 < m["lead_knee_flexion_fp_deg"] < 70
    assert 0 < m["trunk_flexion_br_deg"] < 60  # leaning toward the plate at release
    assert 1 < m["pelvis_peak_speed_ms"] < m["shoulder_peak_speed_ms"] < m["wrist_peak_speed_ms"]
    assert 10 < m["wrist_peak_speed_ms"] < 40 and m["wrist_peak_speed_ms"] > 4 * m["pelvis_peak_speed_ms"]
    if fps < 60:
        return  # no timing norms below 60 fps
    # Our FP (lead ankle stops) is ~12 ms after OBP's force-plate FP and our release (wrist speed
    # peak) ~12 ms before OBP's ball release, so FP->BR is ~25 ms shorter than force-plate values.
    assert 80 < m["fp_to_br_ms"] < 200
    for gap in ("gap_pelvis_shoulder_ms", "gap_shoulder_elbow_ms", "gap_elbow_wrist_ms"):
        assert m[gap] > 0, gap  # proximal-to-distal sequence on average


def test_packaged_norms_peak_speeds_fall_with_fps():
    """Lower frame rates smooth harder and sample peaks coarser: lower peak speeds."""
    wrist = [json.loads(PACKAGED[f].read_text(encoding="utf-8"))["norms"]["wrist_peak_speed_ms"]["mean"]
             for f in PACKAGED_FPS]
    assert wrist == sorted(wrist)


# --- regression on the real OBP reference trials (local, git-ignored data) ---

OBP_ZIP = ENGINE / "data" / "obp" / "raw" / "pitching_landmarks.zip"
OBP_META = ENGINE / "data" / "obp" / "repo" / "baseball_pitching" / "data" / "metadata.csv"


@pytest.mark.slow
def test_fps_matched_norms_keep_reference_finding_rates():
    """The reference pitchers analyzed as 60 or 120 fps videos get arm_speed_loss about as often
    as at 240 fps (with 240 Hz norms at 60 fps it was 78 % vs 17 %), and their median wrist-speed
    z stays near 0. Scoring goes through analyze.score_pitch, the pipeline's own code."""
    if not (OBP_ZIP.is_file() and OBP_META.is_file()):
        pytest.skip("local OBP data not present (see engine/data/obp/README.md)")
    from pitchcoach.analyze import score_pitch
    from pitchcoach.diagnosis import diagnose, load_rules
    from pitchcoach.events import throw_direction

    infos = bon.read_trial_info(OBP_META, OBP_META.parent / "poi" / "poi_metrics.csv")
    rules = load_rules()
    rates = (240.0, 120.0, 60.0)
    hits, wrist_z, n = dict.fromkeys(rates, 0), {f: [] for f in rates}, 0
    for trial in bon.read_landmarks(OBP_ZIP):
        info = infos[trial.session_pitch]
        if not bon.in_population(info.level):
            continue
        n += 1
        subject = Subject(info.height_cm, info.hand, "arm")
        for fps in rates:
            kp = bon.trial_keypoints(trial.points, trial.names, bon.OBP_FPS, info.hand, fps)
            events, _ = detect_events(kp, fps, bon.PPM, subject)
            direction = throw_direction(kp, info.hand, events)
            metrics, energy, norm_fps = score_pitch(kp, events, fps, bon.PPM, subject, direction)
            assert norm_fps == fps
            findings = diagnose(metrics, energy.leaks, rules)
            hits[fps] += any(f.issue_id == "arm_speed_loss" for f in findings)
            z = next(m.z for m in metrics if m.name == "wrist_peak_speed_ms")
            if z is not None:
                wrist_z[fps].append(z)
    share = {f: hits[f] / n for f in rates}
    median_z = {f: float(np.median(wrist_z[f])) for f in rates}
    print(f"\nreference trials: {n}; arm_speed_loss share {share}; median wrist z {median_z}")
    assert n >= 300
    for fps in (120.0, 60.0):
        assert share[fps] <= 1.5 * share[240.0] + 0.05, share
        assert share[240.0] <= 1.5 * share[fps] + 0.05, share
        assert abs(median_z[fps]) < 0.25, median_z
