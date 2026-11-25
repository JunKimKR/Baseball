"""Tests for scripts/eval_events.py (pure evaluate(); no MediaPipe)."""

import importlib.util
from pathlib import Path

import pytest

from pitchcoach.models import Events

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_events.py"
spec = importlib.util.spec_from_file_location("eval_events", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

HEADER = "video,fps,height_cm,hand,leg_lift_peak,foot_plant,release\n"


def write(tmp_path, rows):
    p = tmp_path / "labels.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return p


def test_tolerance_scales_with_fps():
    assert ev.tolerance_frames(120) == 2
    assert ev.tolerance_frames(240) == 4
    assert ev.tolerance_frames(60) == 1


def test_evaluate_ratio(tmp_path):
    p = write(tmp_path, ["a.mp4,120,180,R,100,150,200", "b.mp4,240,180,R,100,150,200"])
    results = {
        "a.mp4": Events(leg_lift_peak=102, foot_plant=153, release=198),   # hit, miss, hit
        "b.mp4": Events(leg_lift_peak=104, foot_plant=155, release=200),   # hit (4), miss, hit
    }
    out = ev.evaluate(p, results)
    assert out["per_event"]["leg_lift_peak"]["ratio"] == 1.0
    assert out["per_event"]["foot_plant"]["hits"] == 0
    assert out["per_event"]["release"]["ratio"] == 1.0
    assert out["overall"]["hits"] == 4 and out["overall"]["total"] == 6
    assert out["overall"]["ratio"] == pytest.approx(4 / 6)


def test_empty_label_excluded_and_none_is_miss(tmp_path):
    p = write(tmp_path, ["a.mp4,120,180,R,100,,200", "b.mp4,120,180,L,100,150,200"])
    results = {
        "a.mp4": Events(leg_lift_peak=100, foot_plant=None, release=None),
        "b.mp4": Events(leg_lift_peak=100, foot_plant=None, release=200),
    }
    out = ev.evaluate(p, results)
    assert out["per_event"]["foot_plant"]["total"] == 1      # a.mp4 unlabeled -> excluded
    assert out["per_event"]["foot_plant"]["hits"] == 0       # b.mp4 None -> miss
    assert out["per_event"]["release"]["total"] == 2
    assert out["per_event"]["release"]["hits"] == 1          # a.mp4 None -> miss


def test_comment_rows_ignored_and_no_labels(tmp_path):
    p = write(tmp_path, ["# x.mp4,120,180,R,1,2,3"])
    assert ev.read_labels(p) == []
    assert ev.evaluate(p, {})["overall"]["total"] == 0


def test_main_exit_code(tmp_path, monkeypatch):
    p = write(tmp_path, ["a.mp4,120,180,R,100,150,200"])
    monkeypatch.setattr(ev, "run_analysis", lambda rows, base: ({"a.mp4": Events(100, 150, None, 200)}, {}))
    assert ev.main([str(p)]) == 0
    monkeypatch.setattr(ev, "run_analysis", lambda rows, base: ({"a.mp4": Events(0, 0, None, 0)}, {}))
    assert ev.main([str(p)]) == 1


def test_read_labels_with_bom(tmp_path):
    """Excel saves "UTF-8 CSV" with a byte-order mark; the header must still read as `video`."""
    p = tmp_path / "labels.csv"
    p.write_bytes(("﻿" + HEADER + "a.mp4,120,180,R,100,150,200\n").encode("utf-8"))
    (row,) = ev.read_labels(p)
    assert row["video"] == "a.mp4" and row["labels"]["release"] == 200


def test_run_analysis_records_unreadable_clip_and_continues(tmp_path, monkeypatch):
    import pitchcoach.analyze as analyze_mod

    def fake_analyze(path, subject):
        if path.name == "bad.mp4":
            raise ValueError(f"cannot open video: {path}")
        return type("R", (), {"events": Events(100, 150, None, 200)})()

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)
    rows = ev.read_labels(write(tmp_path, ["bad.mp4,120,180,R,100,150,200",
                                           "good.mp4,120,180,R,100,150,200"]))
    results, errors = ev.run_analysis(rows, tmp_path)
    assert list(errors) == ["bad.mp4"] and "cannot open video" in errors["bad.mp4"]
    assert results["bad.mp4"] == Events() and results["good.mp4"].release == 200


def test_main_reports_error_rows_as_misses(tmp_path, monkeypatch, capsys):
    p = write(tmp_path, ["bad.mp4,120,180,R,100,150,200", "good.mp4,120,180,R,100,150,200"])
    monkeypatch.setattr(ev, "run_analysis", lambda rows, base: (
        {"bad.mp4": Events(), "good.mp4": Events(100, 150, None, 200)},
        {"bad.mp4": "cannot open video"}))
    assert ev.main([str(p)]) == 1  # 3 of 6 labeled events hit
    out = capsys.readouterr().out
    assert "bad.mp4" in out and "cannot open video" in out and "1 clip" in out
    assert "3/6" in out.replace(" ", "")
