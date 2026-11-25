import numpy as np

from pitchcoach.joints import idx
from pitchcoach.models import Keypoints
from pitchcoach.scale import pixels_per_meter


def test_ppm_from_standing_pose():
    xy = np.zeros((20, 33, 2))
    xy[:, idx("left_ankle")] = [100, 1000]
    xy[:, idx("right_ankle")] = [120, 1000]
    xy[:, idx("nose")] = [110, 110]  # 890 px above ankle midpoint
    kp = Keypoints(xy=xy, vis=np.ones((20, 33)))
    assert abs(pixels_per_meter(kp, 100.0) - 1000.0) < 1.0
