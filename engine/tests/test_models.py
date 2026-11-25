import json

import numpy as np

from pitchcoach.joints import JOINTS, idx, side_joints
from pitchcoach.models import (
    AnalysisResult,
    Energy,
    Events,
    Metric,
    Subject,
    VideoMeta,
    Keypoints,
)

SCHEMA_KEYS = {
    "schema_version", "meta", "subject", "warnings", "throw_direction", "keypoints",
    "events", "phases", "metrics", "segments", "energy", "findings",
}


def test_joints_count_and_idx():
    assert len(JOINTS) == 33
    assert JOINTS[0] == "nose" and JOINTS[-1] == "right_foot_index"
    assert idx("right_wrist") == 16


def test_side_joints_right_hand():
    sj = side_joints("R")
    assert sj["throw_wrist"] == idx("right_wrist")
    assert sj["lead_knee"] == idx("left_knee")
    assert sj["rear_hip"] == idx("right_hip")


def test_side_joints_left_hand():
    sj = side_joints("L")
    assert sj["throw_wrist"] == idx("left_wrist")
    assert sj["lead_knee"] == idx("right_knee")
    assert sj["rear_hip"] == idx("left_hip")


def _minimal_result():
    xy = np.zeros((2, 33, 2))
    xy[0, 0, 0] = np.nan
    vis = np.ones((2, 33))
    energy = Energy(
        segments=["legs", "pelvis", "trunk", "arm"],
        intensity=np.array([[0.1, np.nan, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        leaks=[],
        peaks={"pelvis_center": None, "shoulder_center": None, "elbow": None, "wrist": None},
    )
    return AnalysisResult(
        meta=VideoMeta(fps=30.0, width=720, height=1280, n_frames=2),
        subject=Subject(height_cm=185.0, hand="R", camera_side="arm"),
        throw_direction=1,
        keypoints=Keypoints(xy=xy, vis=vis),
        events=Events(None, None, None, None),
        metrics=[Metric("m", "M", float("nan"), "%", "low", "legs")],
        energy=energy,
    )


def test_result_to_dict_schema():
    d = _minimal_result().to_dict()
    assert set(d) == SCHEMA_KEYS
    assert d["schema_version"] == 1
    assert d["meta"]["duration_s"] == 2 / 30.0
    assert d["keypoints"]["joints"] == JOINTS
    assert d["keypoints"]["frames"][0][0] == [None, 0.0, 1.0]
    assert d["metrics"][0]["value"] is None
    assert d["metrics"][0]["norm"] is None
    assert d["energy"]["intensity"][0][1] is None
    assert set(d["energy"]) == {"segments", "intensity", "leaks", "peaks"}
    assert set(d["events"]) == {"leg_lift_peak", "foot_plant", "mer_est", "release"}
    json.dumps(d, allow_nan=False)


def test_self_test_source_serialized():
    from pitchcoach.models import (AnalysisResult, Cause, Finding, SelfTest, Subject, VideoMeta)
    st = SelfTest("t", "do", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}], "b", "Src, 2020")
    res = AnalysisResult(meta=VideoMeta(30.0, 10, 10, 3), subject=Subject(180.0, "R", "arm"),
                         findings=[Finding("i", "t", "e", [Cause("c", "c", st)])])
    assert res.to_dict()["findings"][0]["causes"][0]["self_test"]["source"] == "Src, 2020"


def test_meta_capture_fps_defaults_to_container_fps():
    meta = VideoMeta(fps=30.0, width=10, height=10, n_frames=60)
    assert meta.analysis_fps == 30.0
    res = AnalysisResult(meta=meta, subject=Subject(180.0, "R", "arm"))
    assert res.to_dict()["meta"]["capture_fps"] == 30.0
    slowmo = VideoMeta(fps=30.0, width=10, height=10, n_frames=60, capture_fps=240.0)
    assert slowmo.analysis_fps == 240.0 and slowmo.duration_s == 2.0  # playback duration
    d = AnalysisResult(meta=slowmo, subject=Subject(180.0, "R", "arm")).to_dict()["meta"]
    assert d["fps"] == 30.0 and d["capture_fps"] == 240.0 and d["norm_fps"] is None
