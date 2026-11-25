"""Local FastAPI server: upload a video, poll the analysis, stream the original video."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..analyze import analyze
from ..models import Subject

ALLOWED_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MSG_UNREADABLE = "영상을 읽을 수 없습니다. 다른 파일로 시도해 주세요."


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, ValueError):
        return _MSG_UNREADABLE
    return f"분석 중 오류가 발생했습니다: {type(exc).__name__}"


def create_app(data_dir: Path, estimator=None, web_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="pitchcoach")

    def adir(aid: str) -> Path:
        if not _ID_RE.match(aid):
            raise HTTPException(404, "not found")
        d = data_dir / aid
        if not (d / "status.json").is_file():
            raise HTTPException(404, "not found")
        return d

    def set_status(d: Path, status: str, error: str | None = None) -> None:
        st = _read_json(d / "status.json")
        st.update(status=status, error=error)
        _write_json(d / "status.json", st)

    def run_job(aid: str, video: Path, subject: Subject, capture_fps: float | None) -> None:
        d = data_dir / aid
        try:
            result = analyze(video, subject, estimator, capture_fps=capture_fps)
            _write_json(d / "result.json", result.to_dict())
            set_status(d, "done")
        except Exception as exc:  # never leave the analysis stuck in "processing"
            traceback.print_exc(file=sys.stderr)
            set_status(d, "failed", _error_message(exc))

    @app.post("/api/analyses", status_code=201)
    async def create_analysis(
        background: BackgroundTasks,
        video: UploadFile = File(...),
        height_cm: float = Form(...),
        hand: str = Form(...),
        camera_side: str = Form("arm"),
        capture_fps: float | None = Form(None),
    ):
        if hand not in ("R", "L"):
            raise HTTPException(422, "hand must be R or L")
        if camera_side not in ("arm", "glove"):
            raise HTTPException(422, "camera_side must be arm or glove")
        if not (100.0 <= height_cm <= 250.0):  # also rejects NaN
            raise HTTPException(422, "height_cm must be between 100 and 250")
        if capture_fps is not None and not (1.0 <= capture_fps <= 1000.0):  # also rejects NaN
            raise HTTPException(422, "capture_fps must be between 1 and 1000")
        ext = Path(video.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(422, f"unsupported video type; allowed: {sorted(ALLOWED_EXTS)}")
        data = await video.read()
        if not data:
            raise HTTPException(422, "empty video file")

        aid = uuid.uuid4().hex
        d = data_dir / aid
        d.mkdir(parents=True)
        video_path = d / f"video{ext}"
        video_path.write_bytes(data)
        subject = Subject(height_cm=height_cm, hand=hand, camera_side=camera_side)
        _write_json(d / "status.json", {
            "id": aid,
            "status": "processing",
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": {"height_cm": height_cm, "hand": hand, "camera_side": camera_side},
            "capture_fps": capture_fps,
        })
        background.add_task(run_job, aid, video_path, subject, capture_fps)
        return {"id": aid}

    @app.get("/api/analyses")
    def list_analyses():
        rows = []
        if data_dir.is_dir():
            for d in data_dir.iterdir():
                try:
                    if _ID_RE.match(d.name):
                        st = _read_json(d / "status.json")
                        rows.append({"id": st["id"], "created_at": st["created_at"],
                                     "status": st["status"]})
                except (OSError, ValueError, KeyError):
                    continue
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows

    @app.get("/api/analyses/{aid}")
    def get_analysis(aid: str):
        d = adir(aid)
        st = _read_json(d / "status.json")
        result = None
        if st["status"] == "done" and (d / "result.json").is_file():
            result = _read_json(d / "result.json")
        return {"id": aid, "status": st["status"], "error": st["error"], "result": result}

    @app.get("/api/analyses/{aid}/video")
    def get_video(aid: str):
        d = adir(aid)
        for p in d.glob("video.*"):
            return FileResponse(p)  # Starlette's FileResponse handles Range (206)
        raise HTTPException(404, "not found")

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
