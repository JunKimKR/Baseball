import numpy as np
import pytest

from pitchcoach.metrics import METRIC_DEFS, compute_metrics
from pitchcoach.models import Events, Subject
from pitchcoach.norms import load_norms, score_metrics
from synth import make_pitch

PPM = 500.0
SUBJ = Subject(height_cm=185.0, hand="R", camera_side="arm")
TIMING = ["fp_to_br_ms", "gap_pelvis_shoulder_ms", "gap_shoulder_elbow_ms", "gap_elbow_wrist_ms"]


def _run(fps=240, events=None, camera_moving=False, scale_uncertain=False):
    kp, ev = make_pitch(fps=fps)
    return {m.name: m for m in compute_metrics(kp, events or ev, float(fps), PPM, SUBJ, 1, camera_moving,
                                               scale_uncertain=scale_uncertain)}


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
        if x.unit == "m/s" or n == "trunk_flexion_br_deg":
            assert x.confidence == "low", n
        else:
            assert x.confidence == base[n].confidence, n


ANGLES = ["lead_knee_flexion_fp_deg", "lead_knee_extension_deg", "trunk_flexion_br_deg"]


@pytest.mark.parametrize("fps, expected", [(240, "high"), (120, "high"), (100, "medium"),
                                           (60, "medium"), (50, "low"), (30, "low")])
def test_angle_confidence_follows_fps(fps, expected):
    """Angles shift and scatter at low frame rates (OBP trials at 30 Hz vs 240 Hz: knee flexion at
    plant +4 deg, paired sd above 5 deg), so they lose confidence like the timing metrics."""
    m = _run(fps=fps)
    for n in ANGLES:
        assert m[n].confidence == expected, (fps, n)
    for n in ("pelvis_rotation_proxy_deg", "shoulder_rotation_proxy_deg"):
        assert m[n].confidence == "low", (fps, n)


def test_trunk_flexion_low_under_camera_motion():
    """A hand-held camera rolls, which tilts the image vertical the trunk angle is measured from."""
    m = _run(fps=240, camera_moving=True)
    assert m["trunk_flexion_br_deg"].confidence == "low"
    assert m["lead_knee_flexion_fp_deg"].confidence == "high"  # joint angles do not need the vertical


def test_metrics_missing_event():
    kp, ev = make_pitch(fps=240)
    m = _run(events=Events(ev.leg_lift_peak, None, ev.mer_est, ev.release))
    for n in ("stride_length_pct_height", "lead_knee_flexion_fp_deg", "lead_knee_extension_deg",
              "pelvis_rotation_proxy_deg", "shoulder_rotation_proxy_deg", "fp_to_br_ms"):
        assert m[n].value is None, n
    assert m["trunk_flexion_br_deg"].value is not None
    m = _run(events=Events())
    assert all(x.value is None for x in m.values())


def test_elbow_extension_velocity_is_reference_only():
    """The side view inflates elbow extension velocity ~2.5x (forearm foreshortening), so it is
    still computed but low confidence: no packaged norm and no score."""
    kp, ev = make_pitch(fps=240)
    metrics = compute_metrics(kp, ev, 240.0, PPM, SUBJ, 1)
    (elbow,) = [m for m in score_metrics(metrics, load_norms()) if m.name == "elbow_ext_peak_vel_dps"]
    assert elbow.confidence == "low" and elbow.value is not None
    assert elbow.score is None and elbow.z is None and elbow.norm is None


def test_uncertain_scale_lowers_scale_dependent_metrics():
    """Stride (% of height) and m/s depend on the height-based scale; angles and times do not."""
    base, unsure = _run(), _run(scale_uncertain=True)
    for n, x in unsure.items():
        if x.unit == "m/s" or n == "stride_length_pct_height":
            assert x.confidence == "low", n
        else:
            assert x.confidence == base[n].confidence, n
