"""Tests for Skills.candidate_merger."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.candidate_merger import (
    format_merge_summary,
    merge_candidates,
    write_merged,
)


def _write(tmp_path: Path, name: str, rows) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(rows))
    return p


class TestMergeCandidates:
    def test_returns_list(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.2}])
        rows = merge_candidates([p])
        assert isinstance(rows, list)

    def test_single_file(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.3}])
        rows = merge_candidates([p])
        assert len(rows) == 1

    def test_deduplicates_by_tic_id(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.5}])
        p2 = _write(tmp_path, "b.json", [{"tic_id": 1, "best_fpp": 0.3}])
        rows = merge_candidates([p1, p2])
        assert len(rows) == 1

    def test_prefer_best_fpp_keeps_lower(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.5}])
        p2 = _write(tmp_path, "b.json", [{"tic_id": 1, "best_fpp": 0.1}])
        rows = merge_candidates([p1, p2], prefer="best_fpp")
        assert rows[0]["best_fpp"] == pytest.approx(0.1)

    def test_different_tic_ids_merged(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path, "a.json", [{"tic_id": 1, "best_fpp": 0.2}])
        p2 = _write(tmp_path, "b.json", [{"tic_id": 2, "best_fpp": 0.3}])
        rows = merge_candidates([p1, p2])
        assert len(rows) == 2

    def test_sorted_by_fpp_ascending(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.json", [
            {"tic_id": 1, "best_fpp": 0.8},
            {"tic_id": 2, "best_fpp": 0.2},
        ])
        rows = merge_candidates([p])
        assert rows[0]["best_fpp"] <= rows[1]["best_fpp"]

    def test_empty_files(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.json", [])
        rows = merge_candidates([p])
        assert rows == []

    def test_no_tic_id_skipped(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.json", [{"best_fpp": 0.3}])
        rows = merge_candidates([p])
        assert rows == []

    def test_prefer_rank_score(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path, "a.json", [{"tic_id": 1, "rank_score": 0.9, "best_fpp": 0.1}])
        p2 = _write(tmp_path, "b.json", [{"tic_id": 1, "rank_score": 0.5, "best_fpp": 0.05}])
        rows = merge_candidates([p1, p2], prefer="rank_score")
        assert rows[0]["rank_score"] == pytest.approx(0.9)


class TestWriteMerged:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "merged.json"
        write_merged([{"tic_id": 1}], p)
        assert p.exists()

    def test_content_is_json_array(self, tmp_path: Path) -> None:
        p = tmp_path / "merged.json"
        write_merged([{"tic_id": 1}], p)
        data = json.loads(p.read_text())
        assert isinstance(data, list)


class TestFormatMergeSummary:
    def test_returns_string(self) -> None:
        assert isinstance(format_merge_summary([], 0), str)

    def test_contains_source_count(self) -> None:
        text = format_merge_summary([], 3)
        assert "3" in text
