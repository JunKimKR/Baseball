from pathlib import Path

import pytest

from pitchcoach.models import Metric, Norm
from pitchcoach.norms import load_norms, score_metrics

FIX = Path(__file__).parent / "fixtures" / "norms_test.json"


def m(name, value, conf="high"):
    return Metric(name=name, label=name, value=value, unit="deg", confidence=conf, segment="arm")


@pytest.fixture
def norms():
    return load_norms(FIX)


def test_load_norms(norms):
    assert norms["hi"].mean == 10.0 and norms["hi"].sd == 2.0
    assert norms["lo"].direction == "lower" and norms["hi"].source == "t"


def test_load_default_packaged_norms():
    norms = load_norms()
    assert norms
    for name, n in norms.items():
        assert n.source, name
        assert n.direction in ("higher", "lower", "target"), name
        assert n.sd > 0, name


def test_score_higher(norms):
    a, b = score_metrics([m("hi", 10.0), m("hi", 14.0)], norms)
    assert a.score == 50 and a.z == 0.0 and a.norm is norms["hi"]
    assert b.score == 98 and b.z == 2.0


def test_score_target(norms):
    a, b, c = score_metrics([m("tg", 10.0), m("tg", 12.0), m("tg", 8.0)], norms)
    assert a.score == 100 and b.score == 32 and c.score == 32


def test_score_lower(norms):
    a, b = score_metrics([m("lo", 14.0), m("lo", 6.0)], norms)
    assert a.score == 2 and b.score == 98


def test_score_none_for_low_confidence(norms):
    (r,) = score_metrics([m("hi", 14.0, "low")], norms)
    assert r.score is None and r.z is None and r.norm is norms["hi"]


def test_score_none_for_missing_value_and_bad_sd(norms):
    a, b = score_metrics([m("hi", None), m("bad", 5.0)], norms)
    assert a.score is None and a.norm is norms["hi"]
    assert b.score is None and b.z is None and b.norm is norms["bad"]


def test_score_none_without_norm(norms):
    (r,) = score_metrics([m("zzz", 1.0)], norms)
    assert r.score is None and r.z is None and r.norm is None


def test_score_does_not_mutate(norms):
    x = m("hi", 14.0)
    (r,) = score_metrics([x], norms)
    assert x.score is None and x.norm is None and r is not x


@pytest.fixture
def timing_norms():
    # the packaged gap_elbow_wrist_ms norm: narrower than one 240 Hz frame (4.17 ms)
    return {"gap_elbow_wrist_ms": Norm(mean=43.338, sd=3.4493, direction="target", source="t"),
            "wrist_peak_speed_ms": Norm(mean=19.453, sd=1.0886, direction="higher", source="t")}


def _metric(name, value, unit):
    return Metric(name=name, label=name, value=value, unit=unit, confidence="medium", segment="arm")


def test_timing_sd_widened_by_frame_quantum(timing_norms):
    gap = _metric("gap_elbow_wrist_ms", 33.3, "ms")
    (at60,) = score_metrics([gap], timing_norms, fps=60.0)
    # sd_eff = sqrt(3.4493^2 + (1000/60)^2 / 6) = 7.6285 ms -> z = -1.3159 -> target score 19
    assert at60.z == pytest.approx(-1.3159, abs=1e-4) and at60.score == 19
    assert at60.norm is timing_norms["gap_elbow_wrist_ms"] and at60.norm.sd == 3.4493
    (plain,) = score_metrics([gap], timing_norms)
    assert plain.z == pytest.approx(-2.9102, abs=1e-4) and plain.score == 0
    (zero_fps,) = score_metrics([gap], timing_norms, fps=0.0)
    assert zero_fps.score == 0


def test_non_timing_metrics_unaffected_by_fps(timing_norms):
    wrist = _metric("wrist_peak_speed_ms", 18.0, "m/s")
    (at60,) = score_metrics([wrist], timing_norms, fps=60.0)
    (plain,) = score_metrics([wrist], timing_norms)
    assert at60.score == plain.score == 9 and at60.z == plain.z


# --- fps-matched packaged norms ---

from pitchcoach.norms import PACKAGED_NORM_FPS, load_norms_for_fps, select_norm_fps  # noqa: E402


@pytest.mark.parametrize("fps, expected", [
    (240.0, 240.0), (360.0, 240.0), (200.0, 120.0), (120.0, 120.0), (100.0, 60.0),
    (60.0, 60.0), (59.0, 30.0), (50.0, 30.0), (30.0, 30.0),
    (25.0, 30.0), (10.0, 30.0),            # floor: the lowest norms
    (239.76, 240.0), (119.88, 120.0), (59.94, 60.0), (29.97, 30.0),  # NTSC rates
])
def test_select_norm_fps_largest_at_or_below(fps, expected):
    assert select_norm_fps(fps) == expected


@pytest.mark.parametrize("bad", [0.0, -30.0, float("nan"), float("inf")])
def test_select_norm_fps_rejects_bad_fps(bad):
    with pytest.raises(ValueError):
        select_norm_fps(bad)


def test_packaged_norm_files_match_their_fps():
    assert PACKAGED_NORM_FPS == (30.0, 60.0, 120.0, 240.0)
    for fps in PACKAGED_NORM_FPS:
        norms, norm_fps = load_norms_for_fps(fps)
        assert norm_fps == fps and norms
        for name, n in norms.items():
            assert n.source and n.sd > 0, (fps, name)
    # 60 fps video -> the 60 Hz norms, whose wrist speed is lower than at 240 Hz
    n60, _ = load_norms_for_fps(60.0)
    n240, _ = load_norms_for_fps(240.0)
    assert n60["wrist_peak_speed_ms"].mean < n240["wrist_peak_speed_ms"].mean - 1.0
    assert load_norms()["wrist_peak_speed_ms"] == n240["wrist_peak_speed_ms"]  # default: 240 Hz
    # no timing norms at 30 Hz: timing metrics are not computed below 60 fps
    n30, _ = load_norms_for_fps(30.0)
    assert not any(name in n30 for name in ("fp_to_br_ms", "gap_elbow_wrist_ms"))


def test_timing_sd_widened_only_beyond_norm_quantum(timing_norms):
    gap = _metric("gap_elbow_wrist_ms", 33.3, "ms")
    sd = timing_norms["gap_elbow_wrist_ms"].sd
    # same rate as the norms: their sd already holds this quantum -> no widening
    (same,) = score_metrics([gap], timing_norms, fps=60.0, norm_fps=60.0)
    assert same.z == pytest.approx((33.3 - 43.338) / sd)
    # finer video than norms: nothing to add
    (finer,) = score_metrics([gap], timing_norms, fps=100.0, norm_fps=60.0)
    assert finer.z == same.z
    # coarser video than norms: only the extra quantum
    (coarser,) = score_metrics([gap], timing_norms, fps=60.0, norm_fps=240.0)
    extra = ((1000 / 60) ** 2 - (1000 / 240) ** 2) / 6
    assert coarser.z == pytest.approx((33.3 - 43.338) / (sd * sd + extra) ** 0.5)
    assert coarser.norm.sd == sd  # the attached norm stays the population norm
