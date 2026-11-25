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


def test_angle_right_angle():
    from pitchcoach.kinematics import angle
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 0.0]])
    c = np.array([[0.0, 1.0]])
    assert abs(angle(a, b, c)[0] - 90.0) < 0.01
    assert abs(angle(a, b, -a)[0] - 180.0) < 0.01


def test_angular_velocity_constant():
    from pitchcoach.kinematics import angular_velocity
    deg = np.arange(20) * 2.0
    assert np.allclose(angular_velocity(deg, 100.0), 200.0)


def test_rotation_proxy_half_width():
    from pitchcoach.kinematics import rotation_proxy
    left = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    right = np.zeros((3, 2))
    out = rotation_proxy(left, right)
    assert abs(out[1] - 60.0) < 0.1
    assert abs(out[2] - 0.0) < 0.1
    assert abs(out[0] - 90.0) < 0.1


def test_rotation_proxy_all_nan_no_warning():
    from pitchcoach.kinematics import rotation_proxy
    out = rotation_proxy(np.full((3, 2), np.nan), np.zeros((3, 2)))
    assert np.isnan(out).all()


def test_trunk_flexion_sign():
    from pitchcoach.kinematics import trunk_flexion
    hip = np.array([[0.0, 0.0], [0.0, 0.0]])
    sh = np.array([[0.0, -1.0], [1.0, -1.0]])
    assert np.allclose(trunk_flexion(hip, sh, 1), [0.0, 45.0])
    assert np.allclose(trunk_flexion(hip, sh, -1), [0.0, -45.0])


def test_peak_frames_synthetic():
    from pitchcoach.kinematics import peak_frames
    from pitchcoach.models import Subject
    from synth import make_pitch
    kp, ev = make_pitch(fps=240)
    subj = Subject(height_cm=185.0, hand="R", camera_side="arm")
    pk = peak_frames(kp, 240.0, 500.0, subj, ev)
    fp = round(1.25 * 240)
    for key, off in (("pelvis_center", 10), ("shoulder_center", 60), ("elbow", 125), ("wrist", 150)):
        assert abs(pk[key] - (fp + round(off * 0.24))) <= 2, key
    none = peak_frames(kp, 240.0, 500.0, subj, type(ev)())
    assert set(none.values()) == {None}
