import numpy as np

from pitchcoach.models import Keypoints, Subject, VideoMeta
from pitchcoach.quality import check_quality, longest_gap_s

KEY = [11, 13, 15]
SUBJ = Subject(height_cm=180, hand="R", camera_side="arm")


def _kp(n, vis=1.0):
    return Keypoints(xy=np.zeros((n, 33, 2)), vis=np.full((n, 33), vis))


def _codes(meta, kp, subj=SUBJ):
    return [w.code for w in check_quality(meta, kp, subj, KEY)]


def test_quality_clean():
    assert _codes(VideoMeta(240, 1080, 1920, 480), _kp(480)) == []


def test_quality_low_fps():
    assert _codes(VideoMeta(90, 1080, 1920, 180), _kp(180)) == ["LOW_FPS"]
    assert _codes(VideoMeta(50, 1080, 1920, 100), _kp(100)) == ["LOW_FPS", "TIMING_DISABLED"]


def test_quality_fps_boundaries():
    assert _codes(VideoMeta(120, 1080, 1920, 240), _kp(240)) == []
    assert _codes(VideoMeta(60, 1080, 1920, 120), _kp(120)) == ["LOW_FPS"]


def test_quality_glove_side():
    subj = Subject(height_cm=180, hand="R", camera_side="glove")
    assert _codes(VideoMeta(240, 1080, 1920, 480), _kp(480), subj) == ["GLOVE_SIDE_CAMERA"]


def test_quality_flags_long_gap():
    kp = _kp(480)
    kp.vis[100:130, 13] = 0.0
    assert longest_gap_s(kp, 240.0, KEY) == 30 / 240
    assert _codes(VideoMeta(240, 1080, 1920, 480), kp) == ["KEYPOINTS_MISSING"]


def test_quality_rejects_no_person():
    ws = check_quality(VideoMeta(240, 1080, 1920, 480), _kp(480, 0.0), SUBJ, KEY)
    no_person = [w for w in ws if w.code == "NO_PERSON"]
    assert len(no_person) == 1 and no_person[0].severity == "error"


def test_quality_too_short():
    ws = check_quality(VideoMeta(240, 1080, 1920, 120), _kp(120), SUBJ, KEY)
    assert [(w.code, w.severity) for w in ws] == [("TOO_SHORT", "error")]


def test_quality_uses_capture_fps():
    # 240 fps slow motion in a 30 fps container: 480 frames = 2 s of motion (16 s of playback)
    slowmo = VideoMeta(30, 1080, 1920, 480, capture_fps=240.0)
    assert _codes(slowmo, _kp(480)) == []
    # 120 frames at 240 fps is 0.5 s of motion even though it plays for 4 s
    short = VideoMeta(30, 1080, 1920, 120, capture_fps=240.0)
    assert [w.code for w in check_quality(short, _kp(120), SUBJ, KEY)] == ["TOO_SHORT"]
    # and a clip captured at 30 fps but labeled 240 gets the low-fps warnings
    assert _codes(VideoMeta(240, 1080, 1920, 480, capture_fps=30.0), _kp(480)) == ["LOW_FPS", "TIMING_DISABLED"]
