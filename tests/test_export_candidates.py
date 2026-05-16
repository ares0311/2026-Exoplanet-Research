"""Tests for Skills/export_candidates.py."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.export_candidates import (  # noqa: E402
    to_csv,
    to_markdown_table,
    to_summary_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs: object) -> dict:
    base = {
        "candidate_id": "TIC1_01",
        "target_id": "TIC 1",
        "period_days": 5.0,
        "depth_ppm": 1000.0,
        "snr": 10.0,
        "scores": {
            "false_positive_probability": 0.05,
            "detection_confidence": 0.80,
        },
        "provenance_score": 0.90,
        "rank_score": 0.75,
        "pathway": "tfop_ready",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        to_csv([_row()], out)
        assert out.exists()

    def test_csv_has_header_row(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        to_csv([_row()], out)
        with out.open() as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "FPP" in header or "Candidate ID" in header

    def test_csv_row_count_matches_input(self, tmp_path: Path) -> None:
        rows = [_row(candidate_id=f"TIC1_0{i}") for i in range(3)]
        out = tmp_path / "out.csv"
        to_csv(rows, out)
        with out.open() as f:
            lines = list(csv.reader(f))
        assert len(lines) == 4  # 1 header + 3 data

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "out.csv"
        to_csv([_row()], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------


class TestToMarkdownTable:
    def test_returns_string(self) -> None:
        assert isinstance(to_markdown_table([_row()]), str)

    def test_empty_returns_placeholder(self) -> None:
        s = to_markdown_table([])
        assert "No candidates" in s

    def test_contains_pipe_characters(self) -> None:
        s = to_markdown_table([_row()])
        assert "|" in s

    def test_contains_candidate_id(self) -> None:
        s = to_markdown_table([_row(candidate_id="TIC999_01")])
        assert "TIC999_01" in s

    def test_separator_row_present(self) -> None:
        s = to_markdown_table([_row()])
        assert "---" in s


# ---------------------------------------------------------------------------
# to_summary_stats
# ---------------------------------------------------------------------------


class TestToSummaryStats:
    def test_n_candidates_correct(self) -> None:
        rows = [_row(), _row(candidate_id="TIC1_02")]
        stats = to_summary_stats(rows)
        assert stats["n_candidates"] == 2

    def test_mean_fpp_computed(self) -> None:
        r1 = _row()
        r1["scores"]["false_positive_probability"] = 0.10
        r2 = _row(candidate_id="TIC1_02")
        r2["scores"]["false_positive_probability"] = 0.30
        stats = to_summary_stats([r1, r2])
        assert stats["mean_fpp"] == pytest.approx(0.20)

    def test_pathway_counts_aggregated(self) -> None:
        rows = [
            _row(pathway="tfop_ready"),
            _row(candidate_id="TIC1_02", pathway="tfop_ready"),
            _row(candidate_id="TIC1_03", pathway="planet_hunters_discussion"),
        ]
        stats = to_summary_stats(rows)
        assert stats["pathway_counts"]["tfop_ready"] == 2
        assert stats["pathway_counts"]["planet_hunters_discussion"] == 1

    def test_empty_rows_returns_zero_count(self) -> None:
        stats = to_summary_stats([])
        assert stats["n_candidates"] == 0
        assert stats["mean_fpp"] is None
