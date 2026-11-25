import numpy as np

from pitchcoach.models import Keypoints
from pitchcoach.smoothing import smooth


def _kp(xy, vis=None):
    if vis is None:
        vis = np.ones(xy.shape[:2])
    return Keypoints(xy=xy, vis=vis)


def test_smooth_removes_noise():
    rng = np.random.default_rng(0)
    fps, n = 120.0, 240
    t = np.arange(n) / fps
    clean = np.zeros((n, 33, 2))
    clean[:, :, 0] = (100 + 50 * np.sin(2 * np.pi * 1.0 * t))[:, None]
    clean[:, :, 1] = (200 + 30 * np.sin(2 * np.pi * 0.5 * t))[:, None]
    noise = rng.normal(0, 3.0, clean.shape)
    out = smooth(_kp(clean + noise), fps)
    rmse = np.sqrt(np.mean((out.xy - clean) ** 2))
    noise_rmse = np.sqrt(np.mean(noise ** 2))
    assert rmse < 0.5 * noise_rmse


def test_smooth_fills_short_gaps():
    fps, n = 120.0, 100
    xy = np.zeros((n, 33, 2))
    xy[:, :, 0] = np.linspace(0, 99, n)[:, None]
    vis = np.ones((n, 33))
    vis[40:45, 5] = 0.0
    xy[40:45, 5] = 9999.0  # garbage where invisible
    out = smooth(_kp(xy, vis), fps)
    assert not np.isnan(out.xy).any()
    assert np.array_equal(out.vis, vis)
    seg = out.xy[40:45, 5, 0]
    assert np.all(seg > out.xy[39, 5, 0] - 1) and np.all(seg < out.xy[45, 5, 0] + 1)
    assert np.all(seg < 200)


def test_smooth_all_invisible_joint_stays_nan():
    xy = np.random.default_rng(1).normal(size=(60, 33, 2))
    vis = np.ones((60, 33))
    vis[:, 3] = 0.0
    out = smooth(_kp(xy, vis), 60.0)
    assert np.isnan(out.xy[:, 3]).all()
    assert not np.isnan(np.delete(out.xy, 3, axis=1)).any()


def test_smooth_short_clip_does_not_crash():
    xy = np.random.default_rng(2).normal(size=(5, 33, 2))
    out = smooth(_kp(xy), 120.0)
    assert out.xy.shape == xy.shape and not np.isnan(out.xy).any()
