import time
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():  # starlette 1.x nags about httpx (httpx2 not installed)
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

import pitchcoach.analyze as analyze_mod
from pitchcoach.models import Keypoints, VideoMeta
from pitchcoach.server.app import create_app

from synth import make_pitch


class FakeEstimator:
    def __init__(self, kp, meta):
        self.kp, self.meta = kp, meta

    def estimate(self, video_path):
        return self.kp, self.meta


class BoomEstimator:
    def __init__(self, exc):
        self.exc = exc

    def estimate(self, video_path):
        raise self.exc


@pytest.fixture(scope="module")
def pitch():
    kp, _ = make_pitch()
    return kp, VideoMeta(fps=240.0, width=1080, height=1920, n_frames=kp.xy.shape[0])


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    monkeypatch.setattr(analyze_mod, "camera_motion_ratio", lambda p, kp: 0.0)


def _client(tmp_path, est):
    return TestClient(create_app(tmp_path, est))


def _upload(client, name="clip.mp4", data=b"0123456789" * 10, **form):
    fields = {"height_cm": "185", "hand": "R"}
    fields.update(form)
    return client.post("/api/analyses", files={"video": (name, data, "video/mp4")}, data=fields)


def test_upload_then_done(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    r = _upload(c)
    assert r.status_code == 201
    aid = r.json()["id"]
    assert len(aid) == 32
    body = c.get(f"/api/analyses/{aid}").json()
    assert body["status"] == "done" and body["error"] is None
    assert body["result"]["schema_version"] == 1
    assert (tmp_path / aid / "video.mp4").exists()
    assert (tmp_path / aid / "status.json").exists()
    assert (tmp_path / aid / "result.json").exists()


def test_get_unknown_404(tmp_path):
    c = _client(tmp_path, None)
    assert c.get("/api/analyses/" + "a" * 32).status_code == 404
    assert c.get("/api/analyses/" + "a" * 32 + "/video").status_code == 404


def test_path_traversal_id_404(tmp_path):
    c = _client(tmp_path, None)
    assert c.get("/api/analyses/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/analyses/not-hex").status_code == 404
    assert c.get("/api/analyses/not-hex/video").status_code == 404


def test_video_range_request(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    aid = _upload(c).json()["id"]
    r = c.get(f"/api/analyses/{aid}/video", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206
    assert r.content == b"0123456789"
    assert r.headers["content-range"].startswith("bytes 0-9/")


def test_invalid_hand_422(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    assert _upload(c, hand="X").status_code == 422
    assert _upload(c, camera_side="front").status_code == 422


def test_height_out_of_range_422(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    assert _upload(c, height_cm="50").status_code == 422
    assert _upload(c, height_cm="300").status_code == 422
    assert _upload(c, height_cm="abc").status_code == 422


def test_bad_extension_and_empty_file_422(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    assert _upload(c, name="evil.exe").status_code == 422
    assert _upload(c, name="noext").status_code == 422
    assert _upload(c, data=b"").status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_extension_kept_lowercase(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    aid = _upload(c, name="../../My Clip.MOV").json()["id"]
    assert (tmp_path / aid / "video.mov").exists()


def test_analysis_with_error_warning_completes(pitch, tmp_path):
    kp, meta = pitch
    bad = Keypoints(np.full_like(kp.xy, np.nan), np.zeros_like(kp.vis))
    c = _client(tmp_path, FakeEstimator(bad, meta))
    aid = _upload(c).json()["id"]
    body = c.get(f"/api/analyses/{aid}").json()
    assert body["status"] == "done"
    assert any(w["code"] == "NO_PERSON" for w in body["result"]["warnings"])


def test_failed_status_on_exception(tmp_path):
    c = _client(tmp_path, BoomEstimator(ValueError("cannot read")))
    aid = _upload(c).json()["id"]
    body = c.get(f"/api/analyses/{aid}").json()
    assert body["status"] == "failed" and body["result"] is None
    assert "영상" in body["error"]
    c2 = _client(tmp_path, BoomEstimator(RuntimeError("x")))
    aid2 = _upload(c2).json()["id"]
    assert "RuntimeError" in c2.get(f"/api/analyses/{aid2}").json()["error"]


def test_list_analyses_newest_first(pitch, tmp_path):
    c = _client(tmp_path, FakeEstimator(*pitch))
    assert c.get("/api/analyses").json() == []
    a = _upload(c).json()["id"]
    time.sleep(0.01)
    b = _upload(c).json()["id"]
    rows = c.get("/api/analyses").json()
    assert [r["id"] for r in rows] == [b, a]
    assert set(rows[0]) == {"id", "created_at", "status"}


def test_serves_web_dir(pitch, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>hi</html>")
    c = TestClient(create_app(tmp_path / "d", FakeEstimator(*pitch), web_dir=web))
    assert "hi" in c.get("/").text
    assert c.get("/api/analyses").json() == []
