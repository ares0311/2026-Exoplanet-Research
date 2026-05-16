"""Tests for Skills/compare_candidates.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.compare_candidates import (  # noqa: E402
    build_comparison_report,
    load_and_merge,
    write_comparison_report,
)


def _row(**kwargs) -> dict:
    base = {
        "candidate_id": "TIC1_01",
        "target_id": "TIC 1",
        "period_days": 5.0,
        "pathway": "tfop_ready",
        "scores": {"false_positive_probability": 0.10},
        "rank_score": 0.80,
    }
    base.update(kwargs)
    return base


class TestLoadAndMerge:
    def test_loads_and_merges_list_json(self, tmp_path: Path) -> None:
        rows = [_row(), _row(candidate_id="TIC1_02")]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(rows))
        merged = load_and_merge([f])
        assert len(merged) == 2

    def test_handles_single_dict_json(self, tmp_path: Path) -> None:
        f = tmp_path / "single.json"
        f.write_text(json.dumps(_row()))
        merged = load_and_merge([f])
        assert len(merged) == 1

    def test_adds_source_file_key(self, tmp_path: Path) -> None:
        f = tmp_path / "results.json"
        f.write_text(json.dumps([_row()]))
        merged = load_and_merge([f])
        assert "_source_file" in merged[0]
        assert str(f) == merged[0]["_source_file"]


class TestBuildComparisonReport:
    def test_empty_rows_returns_placeholder(self) -> None:
        s = build_comparison_report([])
        assert "No candidates" in s or "_" in s

    def test_single_row_produces_markdown_header(self) -> None:
        s = build_comparison_report([_row()])
        assert "#" in s

    def test_multiple_rows_sorted_by_fpp_ascending(self) -> None:
        rows = [
            _row(candidate_id="A", scores={"false_positive_probability": 0.80}),
            _row(candidate_id="B", scores={"false_positive_probability": 0.10}),
        ]
        s = build_comparison_report(rows, sort_by="false_positive_probability")
        assert s.index("B") < s.index("A")

    def test_sort_by_rank_score(self) -> None:
        rows = [
            _row(candidate_id="Lo", rank_score=0.20),
            _row(candidate_id="Hi", rank_score=0.90),
        ]
        s = build_comparison_report(rows, sort_by="rank_score")
        assert s.index("Hi") < s.index("Lo")

    def test_sort_by_period_days(self) -> None:
        rows = [
            _row(candidate_id="Short", period_days=1.0),
            _row(candidate_id="Long", period_days=100.0),
        ]
        s = build_comparison_report(rows, sort_by="period_days")
        assert s.index("Short") < s.index("Long")

    def test_invalid_sort_by_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            build_comparison_report([_row()], sort_by="not_a_key")

    def test_write_comparison_report_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        write_comparison_report([_row()], out)
        assert out.exists()

    def test_write_comparison_report_returns_path(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        result = write_comparison_report([_row()], out)
        assert result == out
