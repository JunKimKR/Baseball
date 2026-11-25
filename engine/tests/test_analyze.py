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


@pytest.mark.parametrize("fps", [240.0, 120.0, 60.0])
def test_analyze_scores_against_norms_of_the_video_fps(fps, tmp_path):
    from pitchcoach.norms import load_norms_for_fps

    kp, _ = make_pitch(fps=fps)
    meta = VideoMeta(fps=fps, width=1080, height=1920, n_frames=kp.xy.shape[0])
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))
    norms, norm_fps = load_norms_for_fps(fps)
    assert norm_fps == fps and res.meta.norm_fps == fps
    assert res.to_dict()["meta"]["norm_fps"] == fps
    wrist = next(m for m in res.metrics if m.name == "wrist_peak_speed_ms")
    assert wrist.norm == norms["wrist_peak_speed_ms"]


def test_error_result_has_no_norm_fps(pitch, tmp_path):
    kp, meta = pitch
    bad = Keypoints(np.full_like(kp.xy, np.nan), np.zeros_like(kp.vis))
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(bad, meta))
    assert res.to_dict()["meta"]["norm_fps"] is None


def test_capture_fps_analyzes_slow_motion_export(tmp_path):
    """A 240 fps slow-motion clip exported as a 30 fps file: every captured frame is kept but the
    container says 30. With capture_fps=240 the analysis runs at 240; the container fps stays for
    playback sync."""
    kp, truth = make_pitch(fps=240)
    meta = VideoMeta(fps=30.0, width=1080, height=1920, n_frames=kp.xy.shape[0])
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta), capture_fps=240.0)
    codes = [w.code for w in res.warnings]
    assert "LOW_FPS" not in codes and "TIMING_DISABLED" not in codes and "EVENT_NOT_FOUND" not in codes
    for name in ("leg_lift_peak", "foot_plant", "mer_est", "release"):
        assert abs(getattr(res.events, name) - getattr(truth, name)) <= 2, name
    assert res.meta.fps == 30.0 and res.meta.capture_fps == 240.0 and res.meta.norm_fps == 240.0
    d = res.to_dict()["meta"]
    assert (d["fps"], d["capture_fps"], d["norm_fps"]) == (30.0, 240.0, 240.0)
    gap = next(m for m in res.metrics if m.name == "gap_elbow_wrist_ms")
    assert abs(gap.value - 25) <= 9  # ms in capture time, not 8x stretched playback time

    plain = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))  # taken as a real 30 fps clip
    assert {"LOW_FPS", "TIMING_DISABLED"} <= {w.code for w in plain.warnings}
    assert plain.meta.capture_fps is None and plain.to_dict()["meta"]["capture_fps"] == 30.0


def test_cli_capture_fps(pitch, tmp_path):
    kp, meta = pitch
    slowmo = VideoMeta(fps=30.0, width=meta.width, height=meta.height, n_frames=meta.n_frames)
    out = tmp_path / "r.json"
    rc = main(["analyze", str(tmp_path / "x.mp4"), "--height", "185", "--hand", "R",
               "--capture-fps", "240", "--out", str(out)], estimator=FakeEstimator(kp, slowmo))
    assert rc == 0
    m = json.loads(out.read_text())["meta"]
    assert (m["fps"], m["capture_fps"]) == (30.0, 240.0)
    for bad in ("0", "1001", "abc"):
        with pytest.raises(SystemExit):
            main(["analyze", str(tmp_path / "x.mp4"), "--height", "185", "--hand", "R",
                  "--capture-fps", bad, "--out", str(out)], estimator=FakeEstimator(kp, slowmo))


def test_analyze_warns_on_uncertain_scale(tmp_path):
    from pitchcoach.joints import idx

    kp, _ = make_pitch(fps=240)
    ank = (kp.xy[:40, idx("left_ankle")] + kp.xy[:40, idx("right_ankle")]) / 2
    kp.xy[:40, idx("nose")] = ank + 0.8 * (kp.xy[:40, idx("nose")] - ank)  # bent over at the start
    meta = VideoMeta(fps=240.0, width=1080, height=1920, n_frames=kp.xy.shape[0])
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))
    w = [w for w in res.warnings if w.code == "SCALE_UNCERTAIN"]
    assert len(w) == 1 and w[0].severity == "warn" and "똑바로 선 자세" in w[0].message
    by = {m.name: m for m in res.metrics}
    for n in ("stride_length_pct_height", "pelvis_peak_speed_ms", "shoulder_peak_speed_ms",
              "wrist_peak_speed_ms"):
        assert by[n].confidence == "low" and by[n].score is None, n
    assert by["trunk_flexion_br_deg"].score is not None  # angles are scale-free


def test_analyze_no_scale_warning_on_upright_start(pitch, tmp_path):
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(*pitch))
    assert "SCALE_UNCERTAIN" not in [w.code for w in res.warnings]


@pytest.mark.parametrize("bad", [0.0, float("nan")])
def test_analyze_bad_container_fps(bad, tmp_path):
    kp, truth = make_pitch(fps=240)
    meta = VideoMeta(fps=bad, width=1080, height=1920, n_frames=kp.xy.shape[0])
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta))
    assert [(w.code, w.severity) for w in res.warnings] == [("BAD_FPS", "error")]
    assert res.metrics == [] and res.findings == []
    json.dumps(res.to_dict(), allow_nan=False)

    # with the capture fps entered, the analysis runs; playback sync falls back to it
    res = analyze(tmp_path / "x.mp4", SUBJECT, FakeEstimator(kp, meta), capture_fps=240.0)
    assert not any(w.severity == "error" for w in res.warnings)
    assert res.meta.fps == 240.0 and res.meta.capture_fps == 240.0
    info = [w for w in res.warnings if w.code == "PLAYBACK_FPS_ASSUMED"]
    assert len(info) == 1 and info[0].severity == "info"
    assert abs(res.events.release - truth.release) <= 2
