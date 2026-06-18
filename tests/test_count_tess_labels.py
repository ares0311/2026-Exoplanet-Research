"""Tests for Skills/count_tess_labels.py."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from count_tess_labels import _cli, count_labels


def _toi_table(
    *, cp: int = 0, kp: int = 0, fp: int = 0, fa: int = 0
) -> pd.DataFrame:
    dispositions = ["CP"] * cp + ["KP"] * kp + ["FP"] * fp + ["FA"] * fa
    return pd.DataFrame({"TFOPWG Disposition": dispositions})


def _read_logged_row(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tess_label_checks ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return dict(row)


def test_count_labels_positive_class_includes_kp() -> None:
    """CP + KP both count as positive class."""
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        return _toi_table(cp=300, kp=200, fp=100, fa=50)

    result = count_labels(read_table_fn=reader)

    assert result["cp"] == 300
    assert result["kp"] == 200
    assert result["fp"] == 100
    assert result["fa"] == 50
    assert result["positive"] == 500   # CP + KP
    assert result["negative"] == 150   # FP + FA
    assert result["total"] == 650


def test_count_labels_gate_open_when_thresholds_met() -> None:
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        return _toi_table(cp=300, kp=200, fp=1000, fa=600)

    result = count_labels(
        min_total=2000, min_positive=400, read_table_fn=reader
    )

    assert result["positive"] == 500
    assert result["total"] == 2100
    assert result["gate_open"] is True


def test_count_labels_gate_closed_when_total_insufficient() -> None:
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        return _toi_table(cp=400, kp=100, fp=900, fa=50)

    result = count_labels(
        min_total=2000, min_positive=400, read_table_fn=reader
    )

    assert result["positive"] == 500
    assert result["total"] == 1450
    assert result["gate_open"] is False


def test_count_labels_gate_closed_when_positive_insufficient() -> None:
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        return _toi_table(cp=100, kp=100, fp=1500, fa=500)

    result = count_labels(
        min_total=2000, min_positive=400, read_table_fn=reader
    )

    assert result["positive"] == 200
    assert result["total"] == 2200
    assert result["gate_open"] is False


def test_count_labels_uses_injected_table_reader() -> None:
    def reader(source_url: str, timeout_seconds: int) -> pd.DataFrame:
        assert source_url.startswith("https://")
        assert timeout_seconds == 7
        return _toi_table(cp=300, kp=150, fp=200, fa=50)

    result = count_labels(
        min_total=600, min_positive=400, timeout_seconds=7, read_table_fn=reader
    )

    assert result["positive"] == 450
    assert result["total"] == 700
    assert result["gate_open"] is True


def test_cli_writes_success_log_for_open_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    code = _cli(
        ["--min-total", "600", "--min-positive", "400", "--log-db", str(db_path)],
        read_table_fn=lambda _url, _timeout: _toi_table(cp=300, kp=150, fp=200, fa=50),
    )

    row = _read_logged_row(db_path)
    assert code == 0
    assert row["status"] == "success"
    assert row["cp"] == 300
    assert row["kp"] == 150
    assert row["fp"] == 200
    assert row["fa"] == 50
    assert row["positive"] == 450
    assert row["negative"] == 250
    assert row["total"] == 700
    assert row["gate_open"] == 1
    assert row["exit_code"] == 0
    assert row["error_message"] is None


def test_cli_migrates_legacy_log_without_threshold_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE tess_label_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_version INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                source_url TEXT NOT NULL,
                cp INTEGER,
                kp INTEGER,
                fp INTEGER,
                fa INTEGER,
                positive INTEGER,
                negative INTEGER,
                total INTEGER,
                gate_open INTEGER,
                exit_code INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )

    code = _cli(
        ["--min-total", "600", "--min-positive", "400", "--log-db", str(db_path)],
        read_table_fn=lambda _url, _timeout: _toi_table(cp=300, kp=150, fp=200, fa=50),
    )

    row = _read_logged_row(db_path)
    with sqlite3.connect(db_path) as conn:
        columns = {
            str(column[1])
            for column in conn.execute("PRAGMA table_info(tess_label_checks)").fetchall()
        }
    assert code == 0
    assert {"min_total", "min_positive"}.issubset(columns)
    assert row["min_total"] == 600
    assert row["min_positive"] == 400
    assert row["status"] == "success"


def test_cli_writes_success_log_for_closed_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    code = _cli(
        ["--min-total", "2000", "--min-positive", "400", "--log-db", str(db_path)],
        read_table_fn=lambda _url, _timeout: _toi_table(cp=100, kp=100, fp=500, fa=50),
    )

    row = _read_logged_row(db_path)
    assert code == 1
    assert row["status"] == "success"
    assert row["positive"] == 200
    assert row["total"] == 750
    assert row["gate_open"] == 0
    assert row["exit_code"] == 1


def test_cli_writes_error_log_for_fetch_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "tess_label_check.sqlite3"

    def failing_reader(_source_url: str, _timeout_seconds: int) -> pd.DataFrame:
        raise TimeoutError("simulated timeout")

    code = _cli(
        ["--min-total", "2000", "--min-positive", "400", "--log-db", str(db_path)],
        read_table_fn=failing_reader,
    )

    row = _read_logged_row(db_path)
    assert code == 2
    assert row["status"] == "error"
    assert row["positive"] is None
    assert row["gate_open"] is None
    assert row["exit_code"] == 2
    assert "simulated timeout" in str(row["error_message"])
