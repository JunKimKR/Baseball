"""Reference-population norms: loading and 0-100 metric scoring."""

from __future__ import annotations

import dataclasses
import json
import math
from importlib import resources
from pathlib import Path
from statistics import NormalDist

from .models import Metric, Norm

_PHI = NormalDist().cdf

# The packaged norms are built at these frame rates (engine/scripts/build_obp_norms.py --fps):
# a video is compared with norms computed as for a video of (nearly) the same frame rate, since
# smoothing and frame sampling lower peak speeds and shift angles at low frame rates.
PACKAGED_NORM_FPS = (30.0, 60.0, 120.0, 240.0)
DEFAULT_NORM_FPS = 240.0
# NTSC-style rates (29.97, 59.94, 119.88, 239.76) are the nominal rate: 0.1 % slower changes
# nothing measurable, while the next lower norms would be off by up to ~1.5 sd (wrist speed).
NORM_FPS_TOLERANCE = 0.01


def packaged_norms_name(norm_fps: float) -> str:
    return f"norms_side_v1_{norm_fps:g}hz.json"


def select_norm_fps(fps: float) -> float:
    """Frame rate of the packaged norms for a video analyzed at `fps`: the largest norm fps not
    above it (within NORM_FPS_TOLERANCE), and the lowest norms below that."""
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a positive number, got {fps!r}")
    usable = [n for n in PACKAGED_NORM_FPS if n <= fps * (1 + NORM_FPS_TOLERANCE)]
    return max(usable) if usable else PACKAGED_NORM_FPS[0]


def _parse(raw: dict) -> dict[str, Norm]:
    return {
        name: Norm(
            mean=float(e["mean"]),
            sd=float(e["sd"]),
            direction=e["direction"],
            source=e.get("source", ""),
        )
        for name, e in raw["norms"].items()
    }


def _packaged(norm_fps: float) -> dict[str, Norm]:
    name = packaged_norms_name(norm_fps)
    res = resources.files("pitchcoach") / "data" / name
    if not res.is_file():
        raise FileNotFoundError(f"packaged norm file not found: data/{name}")
    raw = json.loads(res.read_text(encoding="utf-8"))
    if raw.get("norm_fps") != norm_fps:
        raise ValueError(f"data/{name} records norm_fps {raw.get('norm_fps')!r}, expected {norm_fps:g}")
    return _parse(raw)


def load_norms(path: Path | None = None) -> dict[str, Norm]:
    """Load norms from `path`, or the packaged 240 Hz norms."""
    if path is None:
        return _packaged(DEFAULT_NORM_FPS)
    return _parse(json.loads(Path(path).read_text(encoding="utf-8")))


def load_norms_for_fps(fps: float) -> tuple[dict[str, Norm], float]:
    """(norms, norm_fps): the packaged norms for a video analyzed at `fps` (select_norm_fps)."""
    norm_fps = select_norm_fps(fps)
    return _packaged(norm_fps), norm_fps


def _score(z: float, direction: str) -> int:
    if direction == "higher":
        s = 100 * _PHI(z)
    elif direction == "lower":
        s = 100 * _PHI(-z)
    elif direction == "target":
        s = 100 * 2 * (1 - _PHI(abs(z)))
    else:
        raise ValueError(f"unknown norm direction: {direction!r}")
    return round(s)


def _effective_sd(sd: float, unit: str, fps: float | None, norm_fps: float | None) -> float:
    """sd for z: timing metrics (unit "ms") are widened by the video's frame quantum beyond the
    quantum the norms already contain."""
    if fps is None or not fps > 0 or unit != "ms":
        return sd
    q = 1000.0 / fps
    q_norm = 1000.0 / norm_fps if norm_fps is not None and norm_fps > 0 else 0.0
    return math.sqrt(sd * sd + max(0.0, q * q - q_norm * q_norm) / 6.0)


def score_metrics(metrics: list[Metric], norms: dict[str, Norm],
                  fps: float | None = None, norm_fps: float | None = None) -> list[Metric]:
    """Return new Metrics with z/score/norm attached; inputs are not mutated.

    A timing value (unit "ms") is the difference of two frame times, each quantized to 1000 / fps
    ms (uniform error, variance q**2 / 12, so q**2 / 6 for the difference). Norms built at
    `norm_fps` already contain their own quantum, so with the video's `fps` (> 0) a timing metric
    is scored against sd_eff = sqrt(sd**2 + max(0, (1000 / fps)**2 - (1000 / norm_fps)**2) / 6):
    widened only by the quantization the video adds beyond the norms' (none when the video is at
    least as fine). With norm_fps None the norms are taken as unquantized (the whole video quantum
    is added); with fps None every metric uses the norm's sd. The score formulas are unchanged,
    and the attached norm stays the population norm.
    """
    out: list[Metric] = []
    for met in metrics:
        norm = norms.get(met.name)
        if (
            norm is None
            or met.value is None
            or met.confidence == "low"
            or not norm.sd > 0
            or not math.isfinite(met.value)
        ):
            out.append(dataclasses.replace(met, score=None, z=None, norm=norm))
            continue
        z = (met.value - norm.mean) / _effective_sd(norm.sd, met.unit, fps, norm_fps)
        out.append(dataclasses.replace(met, score=_score(z, norm.direction), z=z, norm=norm))
    return out
