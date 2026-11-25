import cv2
import numpy as np

from pitchcoach.camera import camera_motion_ratio
from pitchcoach.models import Keypoints


def _video(path, shift_per_frame, n=30, w=320, h=240):
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur((rng.random((h, w + 400)) * 255).astype(np.uint8), (3, 3), 0)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        x = 100 + int(round(i * shift_per_frame))
        vw.write(np.ascontiguousarray(bg[:, x:x + w]))
    vw.release()


def _kp(n=30, w=320, h=240):
    xy = np.zeros((n, 33, 2))
    xy[..., 0] = w / 2
    xy[..., 1] = h / 2
    xy[:, 0] += [0, -20]
    xy[:, 1] += [10, 20]
    return Keypoints(xy, np.ones((n, 33)))


def test_camera_motion_detects_pan(tmp_path):
    p = tmp_path / "pan.mp4"
    _video(p, 4)
    assert camera_motion_ratio(p, _kp()) > 0.05


def test_camera_motion_static_is_zero(tmp_path):
    p = tmp_path / "still.mp4"
    _video(p, 0)
    assert camera_motion_ratio(p, _kp()) < 0.01


def test_camera_motion_unreadable_is_zero(tmp_path):
    assert camera_motion_ratio(tmp_path / "none.mp4", _kp()) == 0.0
