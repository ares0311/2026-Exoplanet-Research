"""Tests for Skills.bulk_priority_update."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.bulk_priority_update import update_priorities


def _write_log(path: Path, entries: dict) -> None:
    data = {
        "last_updated": "2026-01-01T00:00:00+00:00",
        "entries": entries,
    }
    path.write_text(json.dumps(data))


class TestUpdatePriorities:
    def test_returns_summary_dict(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        result = update_priorities(log, priority_fn=lambda e: 0.7)
        assert isinstance(result, dict)

    def test_n_entries_correct(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}, "2": {"priority_score": 0.3}})
        result = update_priorities(log, priority_fn=lambda e: 0.8)
        assert result["n_entries"] == 2

    def test_n_updated_counts_changed_scores(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}, "2": {"priority_score": 0.8}})
        result = update_priorities(log, priority_fn=lambda e: 0.8)
        assert result["n_updated"] == 1  # entry "1" changed, "2" unchanged

    def test_mean_new_score_matches_priority_fn(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.2}, "2": {"priority_score": 0.4}})
        result = update_priorities(log, priority_fn=lambda e: 0.6)
        assert result["mean_new_score"] == pytest.approx(0.6)

    def test_scores_written_to_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.1}})
        update_priorities(log, priority_fn=lambda e: 0.9)
        data = json.loads(log.read_text())
        assert data["entries"]["1"]["priority_score"] == pytest.approx(0.9)

    def test_output_path_defaults_to_log_path(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        result = update_priorities(log, priority_fn=lambda e: 0.7)
        assert Path(result["output_path"]) == log

    def test_custom_output_path(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        out = tmp_path / "updated.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        result = update_priorities(log, priority_fn=lambda e: 0.7, output_path=out)
        assert out.exists()
        assert Path(result["output_path"]) == out

    def test_last_updated_field_refreshed(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        update_priorities(log, priority_fn=lambda e: 0.7)
        data = json.loads(log.read_text())
        assert "last_updated" in data
        assert data["last_updated"] != "2026-01-01T00:00:00+00:00"

    def test_empty_entries_returns_zero_counts(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {})
        result = update_priorities(log, priority_fn=lambda e: 0.5)
        assert result["n_entries"] == 0
        assert result["n_updated"] == 0

    def test_mean_old_score_computed_from_original(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.4}, "2": {"priority_score": 0.6}})
        result = update_priorities(log, priority_fn=lambda e: 0.9)
        assert result["mean_old_score"] == pytest.approx(0.5)

    def test_result_has_all_required_keys(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        result = update_priorities(log, priority_fn=lambda e: 0.5)
        for key in ("n_entries", "n_updated", "mean_old_score", "mean_new_score", "output_path"):
            assert key in result

    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        log = tmp_path / "log.json"
        _write_log(log, {"1": {"priority_score": 0.5}})
        update_priorities(log, priority_fn=lambda e: 0.8)
        assert log.exists()
        data = json.loads(log.read_text())
        assert "entries" in data
