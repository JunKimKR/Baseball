"""Event-accuracy evaluation: detected events vs hand-labeled frames.

Usage: .venv/bin/python engine/scripts/eval_events.py labels.csv

Label CSV columns: video,fps,height_cm,hand,leg_lift_peak,foot_plant,release (frame numbers).
Relative video paths resolve against the CSV's directory. An empty label cell means the event is
not labeled (excluded from that event's denominator); a detected None counts as a miss.
Tolerance = 2 * fps / 120 frames. Exit code 0 iff the overall hit ratio >= 0.80.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pitchcoach.models import Events

EVENT_NAMES = ("leg_lift_peak", "foot_plant", "release")
TARGET_RATIO = 0.80


def tolerance_frames(fps: float) -> float:
    """±2 frames at 120 fps, scaled with the frame rate (240 fps -> 4 frames)."""
    return 2 * fps / 120


def read_labels(labels_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        labels = {n: (int(r[n]) if (r.get(n) or "").strip() else None) for n in EVENT_NAMES}
        rows.append({"video": r["video"].strip(), "fps": float(r["fps"]),
                     "height_cm": float(r["height_cm"]), "hand": r["hand"].strip().upper(),
                     "labels": labels})
    return rows


def evaluate(labels_csv: Path, results: dict[str, Events]) -> dict:
    """Pure comparison of detected events (keyed by the CSV's `video` string) with labels."""
    counts = {n: {"hits": 0, "total": 0} for n in EVENT_NAMES}
    for row in read_labels(labels_csv):
        tol = tolerance_frames(row["fps"])
        detected = results.get(row["video"])
        for name, label in row["labels"].items():
            if label is None:
                continue
            counts[name]["total"] += 1
            got = getattr(detected, name, None) if detected is not None else None
            if got is not None and abs(got - label) <= tol:
                counts[name]["hits"] += 1

    def finish(c: dict) -> dict:
        return {**c, "ratio": c["hits"] / c["total"] if c["total"] else None}

    overall = {"hits": sum(c["hits"] for c in counts.values()),
               "total": sum(c["total"] for c in counts.values())}
    return {"per_event": {n: finish(c) for n, c in counts.items()}, "overall": finish(overall)}


def run_analysis(rows: list[dict], base: Path) -> dict[str, Events]:
    from pitchcoach.analyze import analyze  # real MediaPipe; imported lazily
    from pitchcoach.models import Subject

    results: dict[str, Events] = {}
    for row in rows:
        path = Path(row["video"])
        if not path.is_absolute():
            path = base / path
        subject = Subject(height_cm=row["height_cm"], hand=row["hand"], camera_side="arm")
        print(f"analyzing {path} ...", file=sys.stderr)
        results[row["video"]] = analyze(path, subject).events
    return results


def _fmt(c: dict) -> str:
    ratio = "n/a" if c["ratio"] is None else f"{c['ratio']:.1%}"
    return f"{c['hits']:>4}/{c['total']:<4} {ratio:>7}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels", type=Path)
    args = ap.parse_args(argv)
    rows = read_labels(args.labels)
    results = run_analysis(rows, args.labels.resolve().parent)
    out = evaluate(args.labels, results)
    print(f"{'event':<16}{'hit/total':>10}{'ratio':>8}   (tolerance = 2 x fps / 120 frames)")
    for name, c in out["per_event"].items():
        print(f"{name:<16}{_fmt(c)}")
    print(f"{'overall':<16}{_fmt(out['overall'])}   target >= {TARGET_RATIO:.0%}")
    ratio = out["overall"]["ratio"]
    return 0 if ratio is not None and ratio >= TARGET_RATIO else 1


if __name__ == "__main__":
    raise SystemExit(main())
