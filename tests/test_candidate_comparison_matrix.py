"""Tests for Skills/candidate_comparison_matrix.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_comparison_matrix import (
    build_comparison_matrix,
    format_comparison_matrix,
)


class TestBuildComparisonMatrix:
    def _sample(self) -> list[dict]:
        return [
            {"tic_id": "123", "period_days": 5.0, "depth_ppm": 1000.0,
             "false_positive_probability": 0.05, "detection_confidence": 0.9,
             "pathway": "tfop_ready", "n_transits": 4},
            {"tic_id": "456", "period_days": 12.3, "depth_ppm": 500.0,
             "false_positive_probability": 0.20, "detection_confidence": 0.7,
             "pathway": "planet_hunters_discussion", "n_transits": 2},
        ]

    def test_basic_two_candidates(self) -> None:
        r = build_comparison_matrix(self._sample())
        assert r.n_candidates == 2
        assert r.flag == "OK"

    def test_empty_input(self) -> None:
        r = build_comparison_matrix([])
        assert r.flag == "NO_CANDIDATES"

    def test_column_labels(self) -> None:
        r = build_comparison_matrix(self._sample())
        assert "TIC ID" in r.columns
        assert "FPP" in r.columns

    def test_row_count(self) -> None:
        r = build_comparison_matrix(self._sample())
        assert len(r.rows) == 2

    def test_row_width_matches_columns(self) -> None:
        r = build_comparison_matrix(self._sample())
        for row in r.rows:
            assert len(row) == len(r.columns)

    def test_extra_columns(self) -> None:
        cands = [{"tic_id": "1", "snr": 12.5}]
        r = build_comparison_matrix(cands, extra_columns=["snr"])
        assert "snr" in r.columns

    def test_missing_keys_fill_dash(self) -> None:
        r = build_comparison_matrix([{"tic_id": "1"}])
        assert r.flag == "OK"
        for row in r.rows:
            assert "—" in row or len(row) > 0

    def test_nested_scores_extracted(self) -> None:
        cand = {"tic_id": "1", "scores": {"false_positive_probability": 0.10}}
        r = build_comparison_matrix([cand])
        fpp_col_idx = list(r.columns).index("FPP")
        assert r.rows[0][fpp_col_idx] != "—"

    def test_single_candidate(self) -> None:
        r = build_comparison_matrix([self._sample()[0]])
        assert r.n_candidates == 1

    def test_format_markdown(self) -> None:
        r = build_comparison_matrix(self._sample())
        s = format_comparison_matrix(r)
        assert "|" in s
        assert "TIC" in s

    def test_format_empty(self) -> None:
        r = build_comparison_matrix([])
        s = format_comparison_matrix(r)
        assert "No candidates" in s

    def test_float_formatting(self) -> None:
        cands = [{"tic_id": "1", "period_days": 3.141592653}]
        r = build_comparison_matrix(cands)
        period_col = list(r.columns).index("Period (d)")
        val = r.rows[0][period_col]
        assert "3.14" in val
