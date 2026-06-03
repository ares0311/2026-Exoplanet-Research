"""Tests for Skills/count_tess_labels.py."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from count_tess_labels import _cli, count_labels


def _toi_table(*, cp: int = 0, fp: int = 0, eb: int = 0) -> pd.DataFrame:
    dispositions = ["CP"] * cp + ["FP"] * fp + ["EB"] * eb
    return pd.DataFrame({"TFOPWG Disposition": dispositions})


def _read_logged_row(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tess_label_checks ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return dict(row)


def test_count_labels_uses_injected_table_reader() -> None:
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        assert source_url.startswith("https://")
        assert timeout_seconds == 7
        return _toi_table(cp=3, fp=2, eb=1)

    result = count_labels(threshold=3, timeout_seconds=7, read_table_fn=reader)

    assert result == {
        "cp": 3,
        "fp": 2,
        "eb": 1,
        "total": 6,
        "gate_open": True,
    }


def test_cli_writes_success_log_for_open_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    code = _cli(
        ["--threshold", "3", "--log-db", str(db_path)],
        read_table_fn=lambda _url, _timeout: _toi_table(cp=3, fp=1, eb=1),
    )

    row = _read_logged_row(db_path)
    assert code == 0
    assert row["status"] == "success"
    assert row["cp"] == 3
    assert row["fp"] == 1
    assert row["eb"] == 1
    assert row["total"] == 5
    assert row["gate_open"] == 1
    assert row["exit_code"] == 0
    assert row["error_message"] is None


def test_cli_writes_success_log_for_closed_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    code = _cli(
        ["--threshold", "5", "--log-db", str(db_path)],
        read_table_fn=lambda _url, _timeout: _toi_table(cp=2, fp=4, eb=1),
    )

    row = _read_logged_row(db_path)
    assert code == 1
    assert row["status"] == "success"
    assert row["cp"] == 2
    assert row["gate_open"] == 0
    assert row["exit_code"] == 1


def test_cli_writes_error_log_for_fetch_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    def failing_reader(_source_url: str, _timeout_seconds: int) -> pd.DataFrame:
        raise TimeoutError("simulated timeout")

    code = _cli(
        ["--threshold", "5", "--log-db", str(db_path)],
        read_table_fn=failing_reader,
    )

    row = _read_logged_row(db_path)
    assert code == 2
    assert row["status"] == "error"
    assert row["cp"] is None
    assert row["gate_open"] is None
    assert row["exit_code"] == 2
    assert "simulated timeout" in str(row["error_message"])
