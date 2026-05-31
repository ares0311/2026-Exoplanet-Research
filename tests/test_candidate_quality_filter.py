"""Tests for Skills/candidate_quality_filter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_quality_filter import filter_by_preset, format_filter_result


def _make_row(fpp: float = 0.05, dc: float = 0.90, n_transits: int = 3) -> dict:
    return {
        "tic_id": "123456",
        "false_positive_probability": fpp,
        "detection_confidence": dc,
        "n_transits": n_transits,
    }


class TestCandidateQualityFilter:
    def test_strict_passes_good(self) -> None:
        rows = [_make_row(fpp=0.02, dc=0.95, n_transits=4)]
        r = filter_by_preset(rows, "strict")
        assert r.n_passed == 1
        assert r.flag == "OK"

    def test_strict_rejects_high_fpp(self) -> None:
        rows = [_make_row(fpp=0.10, dc=0.95, n_transits=4)]
        r = filter_by_preset(rows, "strict")
        assert r.n_rejected == 1

    def test_moderate_passes_boundary(self) -> None:
        rows = [_make_row(fpp=0.14, dc=0.81, n_transits=2)]
        r = filter_by_preset(rows, "moderate")
        assert r.n_passed == 1

    def test_loose_allows_more(self) -> None:
        rows = [_make_row(fpp=0.25, dc=0.72, n_transits=2)]
        r_strict = filter_by_preset(rows, "strict")
        r_loose = filter_by_preset(rows, "loose")
        assert r_loose.n_passed >= r_strict.n_passed

    def test_unknown_preset(self) -> None:
        r = filter_by_preset([], "nonexistent")
        assert r.flag == "UNKNOWN_PRESET"

    def test_empty_rows(self) -> None:
        r = filter_by_preset([], "moderate")
        assert r.n_input == 0
        assert r.n_passed == 0

    def test_n_input_correct(self) -> None:
        rows = [_make_row(), _make_row(fpp=0.9)]
        r = filter_by_preset(rows, "moderate")
        assert r.n_input == 2

    def test_passed_tic_ids_populated(self) -> None:
        rows = [_make_row(fpp=0.01)]
        r = filter_by_preset(rows, "loose")
        assert "123456" in r.passed_tic_ids

    def test_n_rejected_plus_passed_equals_input(self) -> None:
        rows = [_make_row(fpp=0.01), _make_row(fpp=0.99)]
        r = filter_by_preset(rows, "moderate")
        assert r.n_passed + r.n_rejected == r.n_input

    def test_scores_nested_fpp(self) -> None:
        row = {"tic_id": "999", "scores": {"false_positive_probability": 0.01}}
        r = filter_by_preset([row], "moderate")
        assert r.n_passed == 1

    def test_custom_thresholds(self) -> None:
        rows = [_make_row(fpp=0.20)]
        r = filter_by_preset(rows, "moderate", custom_thresholds={"fpp_max": 0.25})
        assert r.n_passed == 1

    def test_format_returns_string(self) -> None:
        r = filter_by_preset([_make_row()], "moderate")
        s = format_filter_result(r)
        assert isinstance(s, str)
        assert "Preset" in s
