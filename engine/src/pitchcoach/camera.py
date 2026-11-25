"""Camera-motion detection from background feature tracking."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Keypoints

VIS_THRESHOLD = 0.5
BBOX_EXPAND = 0.2
WORK_WIDTH = 480
TARGET_FPS = 30.0


def _bbox(kp: Keypoints, i: int, last):
    ok = (np.asarray(kp.vis[i]) >= VIS_THRESHOLD) & np.isfinite(kp.xy[i]).all(axis=1)
    if not ok.any():
        return last
    pts = kp.xy[i][ok]
    return pts.min(axis=0), pts.max(axis=0)


def _mask(shape, box, scale):
    mask = np.full(shape, 255, np.uint8)
    if box is not None:
        lo, hi = box
        pad = (hi - lo) * BBOX_EXPAND
        x0, y0 = np.floor((lo - pad) * scale).astype(int)
        x1, y1 = np.ceil((hi + pad) * scale).astype(int)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0
    return mask


def camera_motion_ratio(video_path: Path, kp: Keypoints) -> float:
    """Path length of the background's median motion, divided by the frame width.

    Background features are those outside the person's keypoint bbox (expanded 20%; with no
    visible keypoints in a frame, the last known bbox). Clips above ~30 fps are strided to
    about 30 fps: the per-step median displacements are still summed, so the total path
    length stays comparable between frame rates (only jitter within a stride is ignored).
    Frames are downscaled for speed; the ratio is relative to the frame width, so unaffected.
    Returns 0.0 when the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        stride = max(1, int(round(fps / TARGET_FPS))) if fps > 0 else 1
        total = 0.0
        prev = None  # (gray, mask)
        last_box = None
        i = -1
        width = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride:
                continue
            h, w = frame.shape[:2]
            width = w
            scale = min(1.0, WORK_WIDTH / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i < len(kp.xy):
                last_box = _bbox(kp, i, last_box)
            mask = _mask(gray.shape, last_box, scale)
            if prev is not None:
                pg, pm = prev
                pts = cv2.goodFeaturesToTrack(pg, maxCorners=200, qualityLevel=0.01,
                                              minDistance=8, mask=pm)
                if pts is not None and len(pts) >= 5:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, gray, pts, None)
                    good = st.ravel() == 1
                    # keep features that stay in the background region of both frames
                    if good.any():
                        q = nxt[good].reshape(-1, 2)
                        p = pts[good].reshape(-1, 2)
                        qi = np.clip(q.round().astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
                        inside = mask[qi[:, 1], qi[:, 0]] > 0
                        if inside.sum() >= 5:
                            d = np.linalg.norm(q[inside] - p[inside], axis=1)
                            total += float(np.median(d)) / (scale * 1.0)
            prev = (gray, mask)
        return total / width if width else 0.0
    finally:
        cap.release()
