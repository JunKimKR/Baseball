"""Command line interface: `pitchcoach analyze ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze
from .models import Subject


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitchcoach")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze", help="analyze a side-view pitching video")
    p.add_argument("video", type=Path)
    p.add_argument("--height", type=float, required=True, help="pitcher height in cm")
    p.add_argument("--hand", choices=["R", "L"], required=True, help="throwing hand")
    p.add_argument("--camera-side", choices=["arm", "glove"], default="arm")
    p.add_argument("--out", type=Path, required=True, help="result JSON path")
    return parser


def main(argv: list[str] | None = None, estimator=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        if estimator is None and not args.video.exists():
            print(f"video not found: {args.video}", file=sys.stderr)
            return 2
        subject = Subject(height_cm=args.height, hand=args.hand, camera_side=args.camera_side)
        try:
            result = analyze(args.video, subject, estimator)
        except ValueError as exc:  # unreadable video
            print(str(exc), file=sys.stderr)
            return 2
        args.out.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        errors = [w for w in result.warnings if w.severity == "error"]
        for w in errors:
            print(f"[{w.code}] {w.message}", file=sys.stderr)
        return 1 if errors else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
