"""Tests for Skills/fetch_nea_koi_lc_index.py (13 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

import fetch_nea_koi_lc_index
from fetch_nea_koi_lc_index import KoiRecord, fetch_koi_lc_index, format_koi_lc_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_CSV = """kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration
10000001,K00001.01,CONFIRMED,2.204735,134.452987,2.12300
10000002,K00002.01,FALSE POSITIVE,5.000000,135.123456,1.50000
10000003,K00003.01,CONFIRMED,10.123456,136.789012,3.25000
"""

_EMPTY_CSV = ""

_COMMENT_CSV = "# comment only\n"

_UNKNOWN_DISP_CSV = """kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration
10000001,K00001.01,CANDIDATE,2.204735,134.452987,2.123
"""


def _make_tap(csv_text: str):
    def _fn(url: str) -> str:
        return csv_text
    return _fn


def _raise_tap(url: str) -> str:
    raise ConnectionError("no network")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.flag == "OK"


def test_n_confirmed():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.n_confirmed == 2


def test_n_fp():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.n_fp == 1


def test_records_count():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert len(result.records) == 3


def test_record_type():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    rec = result.records[0]
    assert isinstance(rec, KoiRecord)
    assert rec.kepoi_name == "K00001.01"
    assert rec.period_days == pytest.approx(2.204735)


def test_fetch_error_flag():
    result = fetch_koi_lc_index(tap_fn=_raise_tap)
    assert result.flag == "FETCH_ERROR"
    assert len(result.records) == 0


def test_empty_csv_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_EMPTY_CSV))
    assert result.flag == "EMPTY"


def test_comment_only_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_COMMENT_CSV))
    assert result.flag == "EMPTY"


def test_unknown_disposition_skipped():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_UNKNOWN_DISP_CSV))
    assert result.flag == "EMPTY"
    assert len(result.records) == 0


def test_max_rows_respected():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV), max_rows=2)
    assert len(result.records) <= 2


def test_disposition_normalised_false_positive():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    fp_recs = [r for r in result.records if r.disposition == "FALSE_POSITIVE"]
    assert len(fp_recs) == 1


def test_formatter_contains_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    text = format_koi_lc_index(result)
    assert "OK" in text
    assert len(text) > 0


def test_formatter_on_error():
    result = fetch_koi_lc_index(tap_fn=_raise_tap)
    text = format_koi_lc_index(result)
    assert "FETCH_ERROR" in text


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


def test_run_report_success_status_on_ok_flag(monkeypatch):
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["path"] = path
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(fetch_nea_koi_lc_index, "run_and_commit_report", fake_commit)

    fetch_nea_koi_lc_index._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=5.0,
        result=result,
        output_paths=("data/koi_lc_index.json",),
        git_run_fn=object(),
    )
    report = captured["report"]
    assert report.script == "fetch_nea_koi_lc_index"
    assert report.status == "success"
    assert report.items_processed == len(result.records)
    assert report.items_written == len(result.records)
    assert "flag=OK" in report.notes
    assert captured["path"].name == "fetch_nea_koi_lc_index.jsonl"
    assert captured["run_fn"] is not None


def test_run_report_failed_status_on_fetch_error(monkeypatch):
    result = fetch_koi_lc_index(tap_fn=_raise_tap)
    assert result.flag == "FETCH_ERROR"
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        return True

    monkeypatch.setattr(fetch_nea_koi_lc_index, "run_and_commit_report", fake_commit)

    fetch_nea_koi_lc_index._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        result=result,
        output_paths=(),
        git_run_fn=None,
    )
    report = captured["report"]
    assert report.status == "failed"
    assert report.items_processed == 0
    assert "flag=FETCH_ERROR" in report.notes


def test_commit_failure_warns_but_does_not_raise(monkeypatch, capsys):
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    monkeypatch.setattr(
        fetch_nea_koi_lc_index, "run_and_commit_report", lambda *a, **k: False
    )

    fetch_nea_koi_lc_index._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        result=result,
        output_paths=(),
        git_run_fn=None,
    )
    assert "Warning" in capsys.readouterr().out


def test_cli_writes_run_report_with_injected_git_runner(tmp_path, monkeypatch):
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    monkeypatch.setattr(
        fetch_nea_koi_lc_index, "fetch_koi_lc_index", lambda **kwargs: result
    )
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(fetch_nea_koi_lc_index, "run_and_commit_report", fake_commit)

    fake_runner = object()
    out = tmp_path / "koi_lc_index.json"
    code = fetch_nea_koi_lc_index._cli(["--output", str(out)], git_run_fn=fake_runner)

    assert code == 0
    assert captured["run_fn"] is fake_runner
    assert captured["report"].items_written == len(result.records)
