"""Tests for Skills/tess_label_check_summary.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from count_tess_labels import write_log_entry
from tess_label_check_summary import (
    _cli,
    build_summary,
    format_summary,
    summary_to_dict,
)


def _write_success(db_path: Path, *, cp: int = 3, total: int = 7) -> None:
    gate_open = total >= 5 and cp >= 5
    write_log_entry(
        db_path,
        started_at="2026-06-03T00:00:00+00:00",
        finished_at="2026-06-03T00:00:01+00:00",
        elapsed_ms=1000,
        source_url="https://example.test/toi.csv",
        min_total=5,
        min_positive=5,
        exit_code=0 if gate_open else 1,
        status="success",
        result={
            "cp": cp,
            "kp": 0,
            "fp": total - cp,
            "fa": 0,
            "positive": cp,
            "negative": total - cp,
            "total": total,
            "gate_open": gate_open,
        },
    )


def test_missing_db_summary_is_safe(tmp_path: Path) -> None:
    summary = build_summary(tmp_path / "missing.sqlite3")

    assert not summary.exists
    assert summary.n_runs == 0
    assert summary.latest_status is None


def test_summary_counts_success_and_error_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "tess_label_check.sqlite3"
    _write_success(db_path, cp=6, total=10)
    write_log_entry(
        db_path,
        started_at="2026-06-03T00:02:00+00:00",
        finished_at="2026-06-03T00:02:02+00:00",
        elapsed_ms=2000,
        source_url="https://example.test/toi.csv",
        min_total=5,
        min_positive=5,
        exit_code=2,
        status="error",
        error_message="TimeoutError: simulated",
    )

    summary = build_summary(db_path)

    assert summary.exists
    assert summary.n_runs == 2
    assert summary.n_success == 1
    assert summary.n_errors == 1
    assert summary.latest_status == "error"
    assert summary.latest_error_message == "TimeoutError: simulated"
    assert summary.last_success_cp == 6
    assert summary.last_success_gate_open is True


def test_summary_to_dict_is_json_serializable(tmp_path: Path) -> None:
    db_path = tmp_path / "tess_label_check.sqlite3"
    _write_success(db_path, cp=2, total=4)

    payload = summary_to_dict(build_summary(db_path))

    assert json.loads(json.dumps(payload))["last_success_cp"] == 2
    assert payload["last_success_gate_open"] is False


def test_format_summary_mentions_latest_and_last_success(tmp_path: Path) -> None:
    db_path = tmp_path / "tess_label_check.sqlite3"
    _write_success(db_path, cp=5, total=8)

    text = format_summary(build_summary(db_path))

    assert "TESS Label-Check Log Summary" in text
    assert "Latest Run" in text
    assert "Last Successful Count" in text
    assert "Gate open: True" in text


def test_cli_prints_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "tess_label_check.sqlite3"
    _write_success(db_path, cp=5, total=9)

    code = _cli(["--log-db", str(db_path), "--json"])

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["n_runs"] == 1


def test_cli_returns_one_when_log_is_missing(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code = _cli(["--log-db", str(tmp_path / "missing.sqlite3")])

    captured = capsys.readouterr()
    assert code == 1
    assert "Exists: no" in captured.out
