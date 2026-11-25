import numpy as np

from pitchcoach.kinematics import speed


def test_speed_constant_motion():
    fps, ppm = 240.0, 500.0
    step = ppm / fps  # pixels per frame for 1 m/s
    frames = np.arange(50)[:, None]
    direction = np.array([0.6, 0.8])  # diagonal, so both axes count
    points = np.array([300.0, 200.0]) + frames * step * direction
    s = speed(points, fps, ppm)
    assert s.shape == (50,)
    assert np.allclose(s, 1.0, atol=0.01)
