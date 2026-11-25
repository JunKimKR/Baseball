import json

import numpy as np
import pytest

import pitchcoach.analyze as analyze_mod
from pitchcoach.analyze import analyze
from pitchcoach.cli import main
from pitchcoach.models import Keypoints, Subject, VideoMeta

from synth import make_pitch

SUBJECT = Subject(185.0, "R", "arm")


class FakeEstimator:
    def __init__(self, kp, meta):
        self.kp, self.meta = kp, meta

    def estimate(self, video_path):
        return self.kp, self.meta


@pytest.fixture(scope="module")
def pitch():
    kp, _ = make_pitch()
    meta = VideoMeta(fps=240.0, width=1080, height=1920, n_frames=kp.xy.shape[0])
    return kp, meta


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    monkeypatch.setattr(analyze_mod, "camera_motion_ratio", lambda p, kp: 0.0)


def test_analyze_with_fake_estimator(pitch, tmp_path):
    kp, meta = pitch
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))
    d = res.to_dict()
    json.dumps(d)
    assert d["schema_version"] == 1
    assert res.events is not None and res.events.release is not None
    assert len(res.metrics) == 15
    assert isinstance(res.findings, list)
    assert len(res.segments) == 4
    assert res.energy.intensity.shape == (kp.xy.shape[0], 4)
    assert res.keypoints is not None
    assert not any(w.severity == "error" for w in res.warnings)


def test_analyze_returns_early_on_error(pitch, tmp_path):
    kp, meta = pitch
    bad = Keypoints(np.full_like(kp.xy, np.nan), np.zeros_like(kp.vis))
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(bad, meta))
    assert any(w.code == "NO_PERSON" for w in res.warnings)
    assert res.metrics == [] and res.findings == [] and res.phases == [] and res.segments == []
    json.dumps(res.to_dict())


def test_analyze_adds_camera_motion_warning(pitch, tmp_path, monkeypatch):
    kp, meta = pitch
    monkeypatch.setattr(analyze_mod, "camera_motion_ratio", lambda p, k: 0.2)
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))
    w = [w for w in res.warnings if w.code == "CAMERA_MOTION"]
    assert len(w) == 1 and w[0].severity == "warn"
    assert "삼각대" in w[0].message
    ms = [m for m in res.metrics if m.unit == "m/s"]
    assert ms and all(m.confidence == "low" for m in ms)


def test_cli_writes_json(pitch, tmp_path):
    kp, meta = pitch
    out = tmp_path / "r.json"
    rc = main(["analyze", str(tmp_path / "x.mp4"), "--height", "185", "--hand", "R",
               "--out", str(out)], estimator=FakeEstimator(kp, meta))
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["subject"] == {"height_cm": 185.0, "hand": "R", "camera_side": "arm"}
    assert len(d["metrics"]) == 15
