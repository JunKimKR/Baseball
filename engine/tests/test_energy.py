import numpy as np

from pitchcoach.energy import (SEGMENTS, detect_leaks, energy_intensity, segment_scores)
from pitchcoach.kinematics import peak_frames
from pitchcoach.models import Metric, Norm, Subject
from pitchcoach.norms import load_norms
from synth import make_pitch

SUBJ = Subject(height_cm=185.0, hand="R", camera_side="arm")
ORDERED = {"pelvis_center": 100, "shoulder_center": 110, "elbow": 120, "wrist": 130}


def _m(name, segment, score=None, z=None, conf="medium"):
    return Metric(name=name, label=name, value=1.0, unit="", confidence=conf,
                  segment=segment, score=score, z=z)


def test_intensity_shape_and_range():
    kp, _ = make_pitch()
    inten = energy_intensity(kp, 240.0, 500.0, SUBJ, 1, load_norms())
    assert inten.shape == (len(kp.xy), 4)
    assert np.isfinite(inten).all()
    assert inten.min() >= 0.0 and inten.max() <= 1.0
    assert SEGMENTS == ["legs", "pelvis", "trunk", "arm"]


def test_intensity_peaks_in_order():
    kp, _ = make_pitch()
    inten = energy_intensity(kp, 240.0, 500.0, SUBJ, 1, load_norms())
    legs, trunk, arm = (int(np.argmax(inten[:, i])) for i in (0, 2, 3))
    assert legs <= trunk <= arm


def test_no_leak_on_ideal_sequence():
    metrics = [_m("pelvis_peak_speed_ms", "legs", 50, 0.0),
               _m("shoulder_peak_speed_ms", "trunk", 50, 0.0),
               _m("wrist_peak_speed_ms", "arm", 50, 0.0)]
    assert detect_leaks(metrics, ORDERED, 240.0) == []


def test_out_of_order_leak():
    kp, ev = make_pitch(peak_offsets_ms={"pelvis": 10, "shoulder": 140, "elbow": 125, "wrist": 150})
    peaks = peak_frames(kp, 240.0, 500.0, SUBJ, ev)
    leaks = detect_leaks([], peaks, 240.0)
    assert len(leaks) == 1
    lk = leaks[0]
    assert lk.segment == "trunk" and lk.kind == "out_of_order"
    assert lk.frame == peaks["shoulder_center"]
    assert 0 < lk.severity <= 1
    assert "ms" in lk.message and "몸통" in lk.message


def test_speed_drop_leak():
    leaks = detect_leaks([_m("wrist_peak_speed_ms", "arm", 2, -2.0)], ORDERED, 240.0)
    assert len(leaks) == 1
    lk = leaks[0]
    assert lk.segment == "arm" and lk.kind == "speed_drop" and lk.frame == 130
    assert abs(lk.severity - 2 / 3) < 1e-6
    assert "-2.0" in lk.message
    assert detect_leaks([_m("wrist_peak_speed_ms", "arm", 40, -1.0)], ORDERED, 240.0) == []
    assert detect_leaks([_m("wrist_peak_speed_ms", "arm", None, None)], ORDERED, 240.0) == []


def test_no_out_of_order_leak_below_60fps():
    bad = {"pelvis_center": 10, "shoulder_center": 30, "elbow": 20, "wrist": 40}
    assert detect_leaks([], bad, 30.0) == []
    assert len(detect_leaks([], bad, 60.0)) == 1


def test_segment_scores_average_and_none():
    metrics = [_m("a", "legs", 80, conf="high"), _m("b", "legs", 61, conf="medium"),
               _m("c", "legs", None, conf="low"),
               _m("d", "arm", None, conf="low"),
               _m("e", "trunk", 70, conf="high")]
    out = {s.segment: s for s in segment_scores(metrics)}
    assert [s.segment for s in segment_scores(metrics)] == SEGMENTS
    assert out["legs"].score == 70 and out["legs"].confidence == "medium"
    assert out["legs"].label == "다리"
    assert out["pelvis"].score is None and out["pelvis"].confidence == "low"
    assert out["trunk"].score == 70 and out["trunk"].confidence == "high"
    assert out["arm"].score is None and out["arm"].confidence == "low"
