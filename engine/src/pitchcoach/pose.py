"""Video reading and MediaPipe pose estimation."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .models import Keypoints, VideoMeta

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_NAME = "pose_landmarker_heavy.task"
N_LANDMARKS = 33


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        # OpenCV applies the container's rotation metadata itself (the default, set
        # explicitly here); frames are therefore never rotated a second time by us.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def read_frames(path: Path) -> Iterator[np.ndarray]:
    """BGR frames, rotation metadata applied."""
    cap = _open(Path(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_meta(path: Path) -> VideoMeta:
    """fps, frame count and the width/height of the frames as read (after rotation)."""
    cap = _open(Path(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"cannot read a frame from: {path}")
        height, width = frame.shape[:2]
        return VideoMeta(fps=fps, width=int(width), height=int(height), n_frames=n_frames)
    finally:
        cap.release()


class PoseEstimator(Protocol):
    def estimate(self, video_path: Path) -> tuple[Keypoints, VideoMeta]: ...


def ensure_model(model_path: Path | None = None) -> Path:
    """Path of the heavy pose model, downloaded once (atomically) to ~/.cache/pitchcoach/."""
    if model_path is not None:
        return Path(model_path)
    path = Path.home() / ".cache" / "pitchcoach" / MODEL_NAME
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(MODEL_URL, timeout=120) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


class MediaPipeEstimator:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path

    def estimate(self, video_path: Path) -> tuple[Keypoints, VideoMeta]:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        meta = read_meta(video_path)
        fps = meta.fps if meta.fps > 0 else 30.0
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(self.model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        xy_rows: list[np.ndarray] = []
        vis_rows: list[np.ndarray] = []
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for i, bgr in enumerate(read_frames(video_path)):
                h, w = bgr.shape[:2]
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                result = landmarker.detect_for_video(image, int(round(i * 1000.0 / fps)))
                xy = np.full((N_LANDMARKS, 2), np.nan)
                vis = np.zeros(N_LANDMARKS)
                if result.pose_landmarks:
                    for j, lm in enumerate(result.pose_landmarks[0][:N_LANDMARKS]):
                        xy[j] = (lm.x * w, lm.y * h)
                        vis[j] = lm.visibility if lm.visibility is not None else 0.0
                xy_rows.append(xy)
                vis_rows.append(vis)
        n = len(xy_rows)
        meta = VideoMeta(fps=meta.fps, width=meta.width, height=meta.height, n_frames=n)
        if n == 0:
            return Keypoints(np.zeros((0, N_LANDMARKS, 2)), np.zeros((0, N_LANDMARKS))), meta
        return Keypoints(np.stack(xy_rows), np.stack(vis_rows)), meta
