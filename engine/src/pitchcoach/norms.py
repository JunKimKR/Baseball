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


def score_metrics(metrics: list[Metric], norms: dict[str, Norm]) -> list[Metric]:
    """Return new Metrics with z/score/norm attached; inputs are not mutated."""
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
        z = (met.value - norm.mean) / norm.sd
        out.append(dataclasses.replace(met, score=_score(z, norm.direction), z=z, norm=norm))
    return out
