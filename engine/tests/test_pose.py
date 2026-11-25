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
