"""Tests for Skills/snippet_batch_progress.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from snippet_batch_progress import format_batch_progress, parse_batch_progress


def _write_checkpoint(tmp_path: Path, completed: list, failed: list) -> Path:
    p = tmp_path / "checkpoint.json"
    p.write_text(json.dumps({"completed_tic_ids": completed, "failed_tic_ids": failed}))
    return p


def _write_output(tmp_path: Path, snippets: list) -> Path:
    p = tmp_path / "output.json"
    p.write_text(json.dumps(snippets))
    return p


def test_missing_file_returns_invalid(tmp_path: Path):
    r = parse_batch_progress(tmp_path / "nope.json")
    assert r.flag == "INVALID"


def test_empty_checkpoint_returns_empty(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, [], [])
    r = parse_batch_progress(cp)
    assert r.flag == "EMPTY"


def test_completed_counted(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1", "TIC2", "TIC3"], [])
    r = parse_batch_progress(cp)
    assert r.n_completed == 3


def test_failed_counted(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, [], ["TIC1", "TIC2"])
    r = parse_batch_progress(cp)
    assert r.n_failed == 2


def test_pct_done_none_when_no_total(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    r = parse_batch_progress(cp)
    assert r.pct_done is None


def test_pct_done_correct_when_given(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1", "TIC2"], [])
    r = parse_batch_progress(cp, total_manifest_size=4)
    assert abs(r.pct_done - 50.0) < 0.01


def test_pct_done_capped_at_100(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1", "TIC2", "TIC3"], [])
    r = parse_batch_progress(cp, total_manifest_size=2)
    assert r.pct_done <= 100.0


def test_reads_output_for_label_counts(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    out = _write_output(tmp_path, [{"label": 1}, {"label": 0}, {"label": 1}])
    r = parse_batch_progress(cp, output_path=out)
    assert r.label_counts.get(1) == 2
    assert r.label_counts.get(0) == 1


def test_missing_output_gives_empty_label_counts(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    r = parse_batch_progress(cp, output_path=tmp_path / "nope.json")
    assert r.label_counts == {}


def test_in_progress_flag(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    r = parse_batch_progress(cp, total_manifest_size=5)
    assert r.flag == "IN_PROGRESS"


def test_ok_flag_when_done(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1", "TIC2", "TIC3"], [])
    r = parse_batch_progress(cp, total_manifest_size=3)
    assert r.flag == "OK"


def test_format_returns_str(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    r = parse_batch_progress(cp)
    assert isinstance(format_batch_progress(r), str)


def test_format_contains_completed(tmp_path: Path):
    cp = _write_checkpoint(tmp_path, ["TIC1"], [])
    r = parse_batch_progress(cp)
    text = format_batch_progress(r)
    assert "completed" in text.lower()
