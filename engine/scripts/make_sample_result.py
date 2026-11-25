"""Generate web/src/fixtures/sample-result.json from the real engine using synthetic keypoints.

Usage: .venv/bin/python engine/scripts/make_sample_result.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine" / "tests"))

import pitchcoach.analyze as analyze_mod  # noqa: E402
from pitchcoach.analyze import analyze  # noqa: E402
from pitchcoach.models import Subject, VideoMeta  # noqa: E402
from synth import make_pitch  # noqa: E402

OUT = ROOT / "web" / "src" / "fixtures" / "sample-result.json"


class FakeEstimator:
    def __init__(self, kp, meta):
        self.kp, self.meta = kp, meta

    def estimate(self, video_path):
        return self.kp, self.meta


def _round(o):
    if isinstance(o, float):
        return round(o, 2)
    if isinstance(o, list):
        return [_round(v) for v in o]
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    return o


# 60 fps, a common phone rate: against the fps-matched norms the synthetic pitch then shows what
# the web tests need to see, i.e. a warning (LOW_FPS), a leak and findings with and without causes.
SAMPLE_FPS = 60


def main() -> None:
    kp, _ = make_pitch(fps=SAMPLE_FPS)
    meta = VideoMeta(fps=float(SAMPLE_FPS), width=1080, height=1920, n_frames=kp.xy.shape[0])
    analyze_mod.camera_motion_ratio = lambda p, k: 0.0  # no real video to inspect
    res = analyze(Path("sample.mp4"), Subject(185.0, "R", "arm"), FakeEstimator(kp, meta))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_round(res.to_dict()), ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
