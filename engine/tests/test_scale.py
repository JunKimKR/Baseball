import numpy as np

from pitchcoach.joints import idx
from pitchcoach.models import Keypoints
from pitchcoach.scale import is_scale_uncertain, pixels_per_meter
from pitchcoach.smoothing import smooth


def test_ppm_from_standing_pose():
    xy = np.zeros((20, 33, 2))
    xy[:, idx("left_ankle")] = [100, 1000]
    xy[:, idx("right_ankle")] = [120, 1000]
    xy[:, idx("nose")] = [110, 110]  # 890 px above ankle midpoint
    kp = Keypoints(xy=xy, vis=np.ones((20, 33)))
    assert abs(pixels_per_meter(kp, 100.0) - 1000.0) < 1.0


def _crouched_start(kp, frames=30, ratio=0.8):
    """Nose pulled toward the ankle midpoint over the first frames (a pitcher bent over at the
    start, or a camera that later zooms in)."""
    xy = kp.xy.copy()
    ank = (xy[:frames, idx("left_ankle")] + xy[:frames, idx("right_ankle")]) / 2
    xy[:frames, idx("nose")] = ank + ratio * (xy[:frames, idx("nose")] - ank)
    return Keypoints(xy=xy, vis=kp.vis)


def test_scale_certain_when_standing_upright_at_start():
    from synth import make_pitch

    for fps in (30, 240):
        kp, _ = make_pitch(fps=fps)
        assert not is_scale_uncertain(smooth(kp, float(fps)))


def test_scale_uncertain_when_start_pose_is_short():
    from synth import make_pitch

    kp, _ = make_pitch(fps=240)
    assert is_scale_uncertain(_crouched_start(kp))
    assert not is_scale_uncertain(_crouched_start(kp, ratio=0.97))


def test_scale_uncertain_without_nose_or_ankles():
    xy = np.full((40, 33, 2), np.nan)
    assert is_scale_uncertain(Keypoints(xy=xy, vis=np.zeros((40, 33))))
