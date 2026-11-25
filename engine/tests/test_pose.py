import subprocess

import cv2
import numpy as np
import pytest

from pitchcoach.pose import read_frames, read_meta


def _write_video(path, w=100, h=200, n=10, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, : w // 2] = (0, 0, 255)
        vw.write(frame)
    vw.release()


def test_read_meta_plain(tmp_path):
    p = tmp_path / "a.mp4"
    _write_video(p)
    meta = read_meta(p)
    assert (meta.width, meta.height, meta.n_frames) == (100, 200, 10)
    assert meta.fps == pytest.approx(30)
    frames = list(read_frames(p))
    assert len(frames) == 10 and frames[0].shape == (200, 100, 3)


def test_read_frames_applies_rotation(tmp_path):
    src = tmp_path / "a.mp4"
    rot = tmp_path / "rot.mp4"
    _write_video(src)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(src),
           "-c", "copy", str(rot)]
    if subprocess.run(cmd).returncode != 0:
        pytest.skip("ffmpeg could not write rotation metadata")
    meta = read_meta(rot)
    assert (meta.width, meta.height) == (200, 100)
    frames = list(read_frames(rot))
    assert frames[0].shape == (100, 200, 3)
    assert frames[0].shape[1] == meta.width and frames[0].shape[0] == meta.height


@pytest.mark.slow
def test_mediapipe_estimator_on_sample():
    from pathlib import Path

    from pitchcoach.pose import MediaPipeEstimator

    sample = Path(__file__).parents[2] / "samples" / "pitch_side_30fps_handheld.mp4"
    if not sample.exists():
        pytest.skip("sample video not present")
    kp, meta = MediaPipeEstimator().estimate(sample)
    assert kp.xy.shape == (meta.n_frames, 33, 2) and kp.vis.shape == (meta.n_frames, 33)


# --- pinned, verified model download (no network: urlopen is faked) ---

import hashlib  # noqa: E402
import io  # noqa: E402
import urllib.request  # noqa: E402

import pitchcoach.pose as pose  # noqa: E402

GOOD, BAD = b"pose model v1", b"something else"


def _fake_urlopen(payload, calls):
    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(payload)
    return urlopen


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(pose, "MODEL_SHA256", hashlib.sha256(GOOD).hexdigest())


def test_model_url_is_versioned_and_hash_recorded():
    assert "/float16/1/" in pose.MODEL_URL and "latest" not in pose.MODEL_URL
    assert pose.MODEL_SHA256 == "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"


def test_ensure_model_keeps_verified_cache(tmp_path, monkeypatch, pinned):
    (tmp_path / pose.MODEL_NAME).write_bytes(GOOD)
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(GOOD, calls))
    assert pose.ensure_model(cache_dir=tmp_path) == tmp_path / pose.MODEL_NAME
    assert calls == []


def test_ensure_model_replaces_cache_that_fails_the_hash(tmp_path, monkeypatch, pinned):
    (tmp_path / pose.MODEL_NAME).write_bytes(BAD)  # e.g. an older "latest" download
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(GOOD, calls))
    path = pose.ensure_model(cache_dir=tmp_path)
    assert calls == [pose.MODEL_URL] and path.read_bytes() == GOOD


def test_ensure_model_rejects_download_with_wrong_hash(tmp_path, monkeypatch, pinned):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(BAD, []))
    with pytest.raises(pose.ModelDownloadError, match="sha256"):
        pose.ensure_model(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # neither the model nor a partial file is left
