"""Command line interface: `pitchcoach analyze ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze
from .models import Subject


def _capture_fps(text: str) -> float:
    try:
        fps = float(text)
    except ValueError:
        fps = float("nan")
    if not 1 <= fps <= 1000:  # also rejects NaN
        raise argparse.ArgumentTypeError("capture fps must be a number between 1 and 1000")
    return fps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitchcoach")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze", help="analyze a side-view pitching video")
    p.add_argument("video", type=Path)
    p.add_argument("--height", type=float, required=True, help="pitcher height in cm")
    p.add_argument("--hand", choices=["R", "L"], required=True, help="throwing hand")
    p.add_argument("--camera-side", choices=["arm", "glove"], default="arm")
    p.add_argument("--capture-fps", type=_capture_fps, default=None,
                   help="frame rate the clip was shot at, if the file says otherwise "
                        "(slow motion exported as 30 fps: e.g. 240)")
    p.add_argument("--out", type=Path, required=True, help="result JSON path")
    sv = sub.add_parser("serve", help="run the local web server (127.0.0.1 only)")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--data-dir", type=Path, default=Path.home() / ".pitchcoach" / "data")
    sv.add_argument("--web-dir", type=Path, default=None, help="built web app (default: <repo>/web/dist)")
    return parser


def main(argv: list[str] | None = None, estimator=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        if estimator is None and not args.video.exists():
            print(f"video not found: {args.video}", file=sys.stderr)
            return 2
        subject = Subject(height_cm=args.height, hand=args.hand, camera_side=args.camera_side)
        try:
            result = analyze(args.video, subject, estimator, capture_fps=args.capture_fps)
        except ValueError as exc:  # unreadable video
            print(str(exc), file=sys.stderr)
            return 2
        args.out.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        errors = [w for w in result.warnings if w.severity == "error"]
        for w in errors:
            print(f"[{w.code}] {w.message}", file=sys.stderr)
        return 1 if errors else 0
    if args.command == "serve":
        import uvicorn

        from .server.app import create_app

        web_dir = args.web_dir
        if web_dir is None:
            default = Path(__file__).resolve().parents[3] / "web" / "dist"
            web_dir = default if default.is_dir() else None
        args.data_dir.mkdir(parents=True, exist_ok=True)
        uvicorn.run(create_app(args.data_dir, estimator, web_dir), host="127.0.0.1", port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
