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
DEFAULT_NORMS_NAME = "norms_side_v1.json"


def load_norms(path: Path | None = None) -> dict[str, Norm]:
    """Load norms from `path`, or the packaged data/norms_side_v1.json."""
    if path is None:
        res = resources.files("pitchcoach") / "data" / DEFAULT_NORMS_NAME
        if not res.is_file():
            raise FileNotFoundError(f"packaged norm file not found: data/{DEFAULT_NORMS_NAME}")
        text = res.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    return {
        name: Norm(
            mean=float(e["mean"]),
            sd=float(e["sd"]),
            direction=e["direction"],
            source=e.get("source", ""),
        )
        for name, e in raw["norms"].items()
    }


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


def _effective_sd(sd: float, unit: str, fps: float | None) -> float:
    """sd for z: timing metrics (unit "ms") are widened by the video's frame quantum."""
    if fps is None or not fps > 0 or unit != "ms":
        return sd
    frame_ms = 1000.0 / fps
    return math.sqrt(sd * sd + frame_ms * frame_ms / 6.0)


def score_metrics(metrics: list[Metric], norms: dict[str, Norm],
                  fps: float | None = None) -> list[Metric]:
    """Return new Metrics with z/score/norm attached; inputs are not mutated.

    With the video's `fps` (> 0), z of a timing metric (unit "ms") uses
    sd_eff = sqrt(sd**2 + (1000 / fps)**2 / 6) instead of sd. A timing value is the difference
    of two frame times, each quantized to 1000 / fps ms (uniform error, variance q**2 / 12),
    so without this a norm narrower than a frame lets one frame of jitter decide the score.
    The score formulas are unchanged, and the attached norm stays the population norm.
    With fps None, every metric uses the norm's sd.
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
        z = (met.value - norm.mean) / _effective_sd(norm.sd, met.unit, fps)
        out.append(dataclasses.replace(met, score=_score(z, norm.direction), z=z, norm=norm))
    return out
