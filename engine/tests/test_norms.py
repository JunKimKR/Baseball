from pathlib import Path

import pytest

from pitchcoach.models import Metric
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


def test_load_default_missing_or_present():
    try:
        n = load_norms(None)
    except FileNotFoundError as e:
        assert "norms_side_v1.json" in str(e)
    else:
        assert isinstance(n, dict)


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
