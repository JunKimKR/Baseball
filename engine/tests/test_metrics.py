import numpy as np

from pitchcoach.metrics import METRIC_DEFS, compute_metrics
from pitchcoach.models import Events, Subject
from synth import make_pitch

PPM = 500.0
SUBJ = Subject(height_cm=185.0, hand="R", camera_side="arm")
TIMING = ["fp_to_br_ms", "gap_pelvis_shoulder_ms", "gap_shoulder_elbow_ms", "gap_elbow_wrist_ms"]


def _run(fps=240, events=None, camera_moving=False):
    kp, ev = make_pitch(fps=fps)
    return {m.name: m for m in compute_metrics(kp, events or ev, float(fps), PPM, SUBJ, 1, camera_moving)}


def test_metrics_names_fixed():
    kp, ev = make_pitch(fps=240)
    out = compute_metrics(kp, ev, 240.0, PPM, SUBJ, 1)
    assert [m.name for m in out] == [d["name"] for d in METRIC_DEFS]
    assert len(out) == 15


def test_metrics_synthetic_gaps():
    m = _run()
    assert abs(m["gap_pelvis_shoulder_ms"].value - 50) <= 9
    assert abs(m["gap_shoulder_elbow_ms"].value - 65) <= 9
    assert abs(m["gap_elbow_wrist_ms"].value - 25) <= 9
    assert abs(m["fp_to_br_ms"].value - 150) <= 9
    assert m["fp_to_br_ms"].confidence == "high"
    assert m["wrist_peak_speed_ms"].value > 15
    assert all(np.isfinite(x.value) for x in m.values() if x.value is not None)
    assert m["stride_length_pct_height"].value > 30
    assert m["pelvis_rotation_proxy_deg"].confidence == "low"


def test_metrics_low_fps_disables_timing():
    m = _run(fps=50)
    for n in TIMING:
        assert m[n].value is None
    for x in m.values():
        if x.unit in ("m/s", "°/s"):
            assert x.confidence == "low", x.name


def test_metrics_medium_fps_lowers_timing():
    m = _run(fps=100)
    assert m["fp_to_br_ms"].confidence == "medium"
    assert m["fp_to_br_ms"].value is not None


def test_metrics_camera_moving_lowers_linear_speeds():
    base, moving = _run(), _run(camera_moving=True)
    for n, x in moving.items():
        if x.unit == "m/s":
            assert x.confidence == "low"
        else:
            assert x.confidence == base[n].confidence


def test_metrics_missing_event():
    kp, ev = make_pitch(fps=240)
    m = _run(events=Events(ev.leg_lift_peak, None, ev.mer_est, ev.release))
    for n in ("stride_length_pct_height", "lead_knee_flexion_fp_deg", "lead_knee_extension_deg",
              "pelvis_rotation_proxy_deg", "shoulder_rotation_proxy_deg", "fp_to_br_ms"):
        assert m[n].value is None, n
    assert m["trunk_flexion_br_deg"].value is not None
    m = _run(events=Events())
    assert all(x.value is None for x in m.values())
