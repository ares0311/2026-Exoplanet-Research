"""Tests for Skills/tess_tce_fetcher.py"""
import sys
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

import tess_tce_fetcher
from tess_tce_fetcher import (
    fetch_tce_table,
    format_tce_summary,
    tce_to_label_rows,
)

_ROWS = [
    {"ticid": 100, "tce_num": 1, "tce_period": 5.0, "tce_time0bt": 1325.0,
     "tce_duration": 2.5, "tce_depth": 10000, "tce_snr": 15.0,
     "tce_disposition": "PC", "sectors": "1,2"},
    {"ticid": 200, "tce_num": 1, "tce_period": 10.0, "tce_time0bt": 1330.0,
     "tce_duration": 3.0, "tce_depth": 5000, "tce_snr": 8.0,
     "tce_disposition": "FP", "sectors": "3"},
    {"ticid": 300, "tce_num": 1, "tce_period": 7.0, "tce_time0bt": 1340.0,
     "tce_duration": 1.5, "tce_depth": 2000, "tce_snr": 6.0,
     "tce_disposition": "ND", "sectors": "1"},
]


def _mock_fetch(url: str) -> list[dict]:
    return _ROWS


def test_fetch_returns_ok():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.flag == "OK"


def test_fetch_counts_pc():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_planet_candidate == 1


def test_fetch_counts_fp():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_false_positive == 1


def test_fetch_counts_nd():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_not_dispositioned == 1


def test_total_matches_sum():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_total == r.n_planet_candidate + r.n_false_positive + r.n_not_dispositioned


def test_disposition_filter():
    r = fetch_tce_table(fetch_fn=_mock_fetch, disposition_filter=["PC"])
    assert r.n_total == 1
    assert r.n_planet_candidate == 1
    assert r.n_false_positive == 0


def test_empty_response():
    r = fetch_tce_table(fetch_fn=lambda url: [])
    assert r.flag == "EMPTY"


def test_invalid_response():
    r = fetch_tce_table(fetch_fn=lambda url: "not-a-list")
    assert r.flag == "INVALID"


def test_network_error():
    def bad_fetch(url: str) -> list[dict]:
        raise ConnectionError("no network")
    r = fetch_tce_table(fetch_fn=bad_fetch)
    assert r.flag == "INVALID"
    assert r.error_message is not None


def test_stale_endpoint_404_reports_unavailable():
    def bad_fetch(url: str) -> list[dict]:
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    r = fetch_tce_table(fetch_fn=bad_fetch)

    assert r.flag == "UNAVAILABLE"
    assert r.error_message is not None
    assert "404" in r.error_message


def test_tce_to_label_rows_pc():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    pc_rows = [row for row in rows if row["label"] == "planet_candidate"]
    assert len(pc_rows) == 1


def test_tce_to_label_rows_fp():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    fp_rows = [row for row in rows if row["label"] == "false_positive"]
    assert len(fp_rows) == 1


def test_tce_to_label_rows_nd_excluded():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    # ND disposition should not produce label rows
    assert len(rows) == 2


def test_format_returns_string():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    s = format_tce_summary(r)
    assert isinstance(s, str)
    assert "TCE" in s


def test_format_gate_fail():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    s = format_tce_summary(r)
    assert "FAIL" in s  # only 2 usable, < 5000


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


def test_run_report_success_status_on_ok_flag(monkeypatch):
    result = fetch_tce_table(fetch_fn=_mock_fetch)
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["path"] = path
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(tess_tce_fetcher, "run_and_commit_report", fake_commit)

    tess_tce_fetcher._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=5.0,
        result=result,
        n_written=2,
        output_paths=("data/tess_tce_labels.json",),
        git_run_fn=object(),
    )
    report = captured["report"]
    assert report.script == "tess_tce_fetcher"
    assert report.status == "success"
    assert report.items_processed == result.n_total
    assert report.items_written == 2
    assert "flag=OK" in report.notes
    assert "pc=1" in report.notes
    assert captured["path"].name == "tess_tce_fetcher.jsonl"
    assert captured["run_fn"] is not None


def test_run_report_failed_status_on_unavailable_flag(monkeypatch):
    def _raise_404(url: str) -> list[dict]:
        raise HTTPError(url, 404, "Not Found", None, None)

    result = fetch_tce_table(fetch_fn=_raise_404)
    assert result.flag == "UNAVAILABLE"
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        return True

    monkeypatch.setattr(tess_tce_fetcher, "run_and_commit_report", fake_commit)

    tess_tce_fetcher._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        result=result,
        n_written=0,
        output_paths=(),
        git_run_fn=None,
    )
    report = captured["report"]
    assert report.status == "failed"
    assert report.items_processed == 0
    assert "flag=UNAVAILABLE" in report.notes
    assert "error=" in report.notes


def test_commit_failure_warns_but_does_not_raise(monkeypatch, capsys):
    result = fetch_tce_table(fetch_fn=_mock_fetch)
    monkeypatch.setattr(
        tess_tce_fetcher, "run_and_commit_report", lambda *a, **k: False
    )

    tess_tce_fetcher._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        result=result,
        n_written=0,
        output_paths=(),
        git_run_fn=None,
    )
    assert "Warning" in capsys.readouterr().out


def test_cli_writes_run_report_with_injected_git_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(tess_tce_fetcher, "fetch_tce_table", lambda **kwargs: fetch_tce_table(
        fetch_fn=_mock_fetch
    ))
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(tess_tce_fetcher, "run_and_commit_report", fake_commit)

    fake_runner = object()
    out = tmp_path / "tce_labels.json"
    code = tess_tce_fetcher._cli(["--output", str(out)], git_run_fn=fake_runner)

    assert code == 0
    assert captured["run_fn"] is fake_runner
    assert captured["report"].status == "success"
    assert captured["report"].items_written == 2


def test_cli_reports_failure_without_writing_output(tmp_path, monkeypatch):
    def _raise_404(url: str) -> list[dict]:
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(
        tess_tce_fetcher, "fetch_tce_table",
        lambda **kwargs: fetch_tce_table(fetch_fn=_raise_404),
    )
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        return True

    monkeypatch.setattr(tess_tce_fetcher, "run_and_commit_report", fake_commit)

    out = tmp_path / "tce_labels.json"
    code = tess_tce_fetcher._cli(["--output", str(out)])

    assert code == 1
    assert not out.exists()
    assert captured["report"].status == "failed"
