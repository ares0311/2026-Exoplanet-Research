"""Tests for Skills.run_summary_exporter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.run_summary_exporter import (
    RunSummary,
    build_run_summary,
    format_run_summary,
    write_run_summary,
)


def _write(tmp_path: Path, name: str, rows) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(rows))
    return p


class TestBuildRunSummary:
    def test_returns_run_summary(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [{"tic_id": 1, "best_fpp": 0.2}])
        s = build_run_summary([p])
        assert isinstance(s, RunSummary)

    def test_n_total_counts_all(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [
            {"tic_id": 1, "best_fpp": 0.1},
            {"tic_id": 2, "best_fpp": 0.8},
        ])
        s = build_run_summary([p])
        assert s.n_total == 2

    def test_n_candidates_filtered_by_fpp(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [
            {"tic_id": 1, "best_fpp": 0.1},
            {"tic_id": 2, "best_fpp": 0.9},
        ])
        s = build_run_summary([p], fpp_threshold=0.5)
        assert s.n_candidates == 1

    def test_n_alert_ready(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [
            {"tic_id": 1, "best_fpp": 0.05},
            {"tic_id": 2, "best_fpp": 0.3},
        ])
        s = build_run_summary([p], alert_fpp_threshold=0.1)
        assert s.n_alert_ready == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [])
        s = build_run_summary([p])
        assert s.n_total == 0

    def test_top_n_respected(self, tmp_path: Path) -> None:
        rows = [{"tic_id": i, "best_fpp": 0.1} for i in range(30)]
        p = _write(tmp_path, "r.json", rows)
        s = build_run_summary([p], top_n=10)
        assert len(s.candidates) <= 10

    def test_generated_at_is_string(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [])
        s = build_run_summary([p])
        assert isinstance(s.generated_at, str)

    def test_multiple_files(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.1}])
        p2 = _write(tmp_path, "b.json", [{"tic_id": 2, "best_fpp": 0.2}])
        s = build_run_summary([p1, p2])
        assert s.n_total == 2


class TestWriteRunSummary:
    def test_creates_markdown_and_json(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [{"tic_id": 1, "best_fpp": 0.1}])
        s = build_run_summary([p])
        paths = write_run_summary(s, tmp_path / "out")
        assert paths["markdown"].exists()
        assert paths["json"].exists()

    def test_json_output_parseable(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [{"tic_id": 1, "best_fpp": 0.1}])
        s = build_run_summary([p])
        paths = write_run_summary(s, tmp_path / "out")
        data = json.loads(paths["json"].read_text())
        assert "n_total" in data


class TestFormatRunSummary:
    def test_returns_string(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [])
        s = build_run_summary([p])
        assert isinstance(format_run_summary(s), str)

    def test_contains_header(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "r.json", [])
        s = build_run_summary([p])
        assert "Pipeline Run Summary" in format_run_summary(s)
