import numpy as np
import pytest
from scipy.special import ndtr

from pitchcoach.events import build_phases, detect_events, throw_direction
from pitchcoach.joints import idx, side_joints
from pitchcoach.kinematics import speed
from pitchcoach.models import Events, Keypoints, Subject
from pitchcoach.scale import pixels_per_meter
from synth import make_pitch

EVENT_NAMES = ("leg_lift_peak", "foot_plant", "mer_est", "release")
PPM = 500.0


def _subject(hand="R"):
    return Subject(height_cm=185.0, hand=hand, camera_side="arm")


def _assert_close(found, truth, tol=2):
    for name in EVENT_NAMES:
        got, want = getattr(found, name), getattr(truth, name)
        assert got is not None, name
        assert abs(got - want) <= tol, (name, got, want)


def _codes(warnings):
    return [w.code for w in warnings]


def test_events_match_synthetic():
    kp, truth = make_pitch(fps=240)
    events, warnings = detect_events(kp, 240.0, PPM, _subject())
    _assert_close(events, truth)
    assert warnings == []


@pytest.mark.parametrize("hand", ["R", "L"])
@pytest.mark.parametrize("direction", [1, -1])
def test_events_all_orientations(hand, direction):
    kp, truth = make_pitch(fps=240, hand=hand, direction=direction)
    events, warnings = detect_events(kp, 240.0, PPM, _subject(hand))
    _assert_close(events, truth)
    assert warnings == []
    assert throw_direction(kp, hand, events) == direction


def test_events_first_pitch_only():
    kp, truth = make_pitch(fps=240, n_pitches=2)
    events, warnings = detect_events(kp, 240.0, PPM, _subject())
    _assert_close(events, truth)
    assert [(w.code, w.severity) for w in warnings] == [("MULTIPLE_PITCHES", "warn")]
    assert warnings[0].message == "두 번째 투구로 보이는 동작이 있어 첫 투구만 분석했습니다."


def test_no_multiple_pitches_warning_below_60fps():
    """Below 60 fps the release peak is smoothed down, so after-pitch motion can pass the
    0.5 x max test; the warning would be a guess."""
    kp, _ = make_pitch(fps=30, n_pitches=2)
    _, warnings = detect_events(kp, 30.0, PPM, _subject())
    assert "MULTIPLE_PITCHES" not in _codes(warnings)
    kp, _ = make_pitch(fps=60, n_pitches=2)
    _, warnings = detect_events(kp, 60.0, PPM, _subject())
    assert "MULTIPLE_PITCHES" in _codes(warnings)


def test_events_walking_after_pitch_no_multiple():
    kp, truth = make_pitch(fps=240, walk_after=True)
    events, warnings = detect_events(kp, 240.0, PPM, _subject())
    _assert_close(events, truth)
    assert "MULTIPLE_PITCHES" not in _codes(warnings)
    assert warnings == []


def test_events_120fps():
    kp, truth = make_pitch(fps=120)
    events, warnings = detect_events(kp, 120.0, PPM, _subject())
    _assert_close(events, truth)
    assert warnings == []


def test_events_no_motion():
    kp, _ = make_pitch(fps=240)
    n = 480
    jitter = np.random.default_rng(0).normal(0.0, 0.3, (n, 33, 2))
    still = Keypoints(xy=np.repeat(kp.xy[:1], n, axis=0) + jitter, vis=np.ones((n, 33)))
    events, warnings = detect_events(still, 240.0, PPM, _subject())
    assert events == Events(None, None, None, None)
    assert [(w.code, w.severity) for w in warnings] == [("EVENT_NOT_FOUND", "warn")] * 4
    for w, name in zip(warnings, ["레그 리프트", "앞발 착지", "외회전", "릴리스"]):
        assert name in w.message


def test_events_missing_foot_plant_skips_dependents():
    kp, truth = make_pitch(fps=240, direction=-1)
    ankle = side_joints("R")["lead_ankle"]
    kp.xy[:, ankle] = kp.xy[0, ankle]  # lead foot never moves -> no plant
    events, warnings = detect_events(kp, 240.0, PPM, _subject())
    assert abs(events.release - truth.release) <= 2
    assert (events.leg_lift_peak, events.foot_plant, events.mer_est) == (None, None, None)
    assert _codes(warnings) == ["EVENT_NOT_FOUND"] * 3
    assert throw_direction(kp, "R", events) == -1  # falls back to wrist x-velocity


def _crafted_clip(fps, n, wrist_peaks=(), lead_ankle_x=None):
    """Still standing pose; the throwing wrist gets Gaussian speed peaks [(time_s, m/s)]
    along x, and the lead ankle x track is replaced if given."""
    sj = side_joints("R")
    xy = np.repeat(make_pitch(fps=fps)[0].xy[:1], n, axis=0)
    t, sigma = np.arange(n) / fps, 0.02
    for time_s, peak in wrist_peaks:
        xy[:, sj["throw_wrist"], 0] += PPM * peak * sigma * np.sqrt(2 * np.pi) * ndtr((t - time_s) / sigma)
    if lead_ankle_x is not None:
        xy[:, sj["lead_ankle"], 0] = lead_ankle_x
    return Keypoints(xy=xy, vis=np.ones((n, 33)))


@pytest.mark.parametrize("gap_s, warned", [(0.8, False), (1.5, True)])
def test_release_is_first_strong_wrist_peak(gap_s, warned):
    # 5 m/s (< 0.5 x max) is ignored; 12 m/s is the release even though 20 m/s comes later.
    kp = _crafted_clip(240, 960, wrist_peaks=[(0.5, 5.0), (1.0, 12.0), (1.0 + gap_s, 20.0)])
    events, warnings = detect_events(kp, 240.0, PPM, _subject())
    assert events.release == 240
    assert ("MULTIPLE_PITCHES" in _codes(warnings)) == warned


def test_foot_plant_needs_three_still_frames():
    n = 480
    steps = np.where(np.arange(n) < 200, 2.0 * PPM / 240, 0.0)  # 2 m/s until frame 200
    steps[181:184] = 0.0  # stutter: speed 0 at frames 181-182 only
    kp = _crafted_clip(240, n, wrist_peaks=[(1.0, 20.0)], lead_ankle_x=400.0 + np.cumsum(steps))
    events, _ = detect_events(kp, 240.0, PPM, _subject())
    assert (events.foot_plant, events.release) == (200, 240)


def test_throw_direction_fallbacks():
    kp, truth = make_pitch(fps=240, direction=-1)
    assert throw_direction(kp, "R", Events(release=truth.release)) == -1
    assert throw_direction(kp, "R", Events()) == 1


def test_phases_order():
    kp, _ = make_pitch(fps=240)
    n = kp.xy.shape[0]
    events, _ = detect_events(kp, 240.0, PPM, _subject())
    phases = build_phases(events, n, 240.0)
    assert [(p.name, p.label) for p in phases] == [
        ("leg_lift", "레그 리프트"),
        ("stride", "스트라이드"),
        ("arm_cocking", "팔 코킹"),
        ("acceleration", "가속·릴리스"),
        ("follow_through", "팔로스루"),
    ]
    assert phases[0].start == 0
    assert all(0 <= p.start <= p.end <= n - 1 for p in phases)
    assert all(a.end == b.start for a, b in zip(phases, phases[1:]))
    assert phases[-1].end == min(events.release + 120, n - 1)


def test_phases_skip_missing_events():
    events = Events(leg_lift_peak=100, foot_plant=None, mer_est=None, release=300)
    phases = build_phases(events, 400, 240.0)
    assert [(p.name, p.start, p.end) for p in phases] == [
        ("leg_lift", 0, 100), ("follow_through", 300, 399)]
    assert build_phases(Events(), 400, 240.0) == []


# --- make_pitch contract (shared by the tests of Tasks 4, 6, 7 and 9) ---

@pytest.mark.parametrize("fps", [30, 50, 120, 240, 360])
def test_make_pitch_contract(fps):
    kp, truth = make_pitch(fps=fps, ppm=1.0)
    n = kp.xy.shape[0]
    assert kp.xy.shape == (n, 33, 2) and np.isfinite(kp.xy).all()
    assert kp.vis.shape == (n, 33) and (kp.vis == 1).all()
    assert n == round((1.40 + 0.6) * fps) + 1  # clip ends 0.6 s after release
    for name, time_s in zip(EVENT_NAMES, (0.5, 1.25, 1.37, 1.40)):  # round(time_s * fps)
        assert abs(getattr(truth, name) - time_s * fps) <= 0.5 + 1e-9, name
    assert pixels_per_meter(kp, 185.0) == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("offsets", [None, {"shoulder": 140}, {"pelvis": 0, "elbow": 110, "wrist": 140}])
def test_make_pitch_peak_points(offsets):
    fps = 240.0
    kp, truth = make_pitch(fps=240, peak_offsets_ms=offsets)
    xy, sj = kp.xy, side_joints("R")
    points = {
        "pelvis": xy[:, [idx("left_hip"), idx("right_hip")]].mean(axis=1),
        "shoulder": xy[:, [idx("left_shoulder"), idx("right_shoulder")]].mean(axis=1),
        "elbow": xy[:, sj["throw_elbow"]],
        "wrist": xy[:, sj["throw_wrist"]],
    }
    expected_ms = {"pelvis": 10, "shoulder": 60, "elbow": 125, "wrist": 150, **(offsets or {})}
    lo, hi = truth.leg_lift_peak, truth.release + 12  # Task 7 window: LLP .. BR + 0.05 s
    for name, pts in points.items():
        s = speed(pts, fps, PPM)
        peak = lo + int(np.argmax(s[lo:hi + 1]))
        assert abs(peak - (truth.foot_plant + round(expected_ms[name] * fps / 1000))) <= 1, name
    wrist = speed(points["wrist"], fps, PPM)
    assert wrist.max() >= 15 and abs(int(np.argmax(wrist)) - truth.release) <= 1
    behind = xy[truth.foot_plant:truth.release + 1, sj["throw_wrist"], 0] \
        - xy[truth.foot_plant:truth.release + 1, sj["throw_shoulder"], 0]
    assert truth.foot_plant + int(np.argmin(behind)) == truth.mer_est
    assert xy[truth.leg_lift_peak, sj["lead_knee"], 1] < xy[truth.leg_lift_peak, sj["lead_hip"], 1]
    ankle = speed(xy[:, sj["lead_ankle"]], fps, PPM)
    assert ankle[:truth.foot_plant].max() >= 2.0
    assert ankle[truth.foot_plant:truth.release].max() < 0.5


def test_make_pitch_walk_contract():
    fps = 240.0
    kp, truth = make_pitch(fps=240, walk_after=True)
    xy, sj = kp.xy, side_joints("R")
    start, end = truth.release + round(0.6 * fps), truth.release + round(2.1 * fps)
    assert xy.shape[0] == end + round(0.6 * fps) + 1
    walk = slice(start, end + 1)
    assert speed(xy[:, sj["throw_wrist"]], fps, PPM)[walk].max() <= 2.0
    for ankle in ("lead_ankle", "rear_ankle"):
        assert speed(xy[:, sj[ankle]], fps, PPM)[walk].max() == pytest.approx(1.4, abs=0.15)
    for knee, hip in (("lead_knee", "lead_hip"), ("rear_knee", "rear_hip")):
        assert (xy[walk, sj[knee], 1] > xy[walk, sj[hip], 1]).all()
