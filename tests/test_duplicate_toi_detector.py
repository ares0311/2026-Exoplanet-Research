"""Tests for Skills.duplicate_toi_detector."""
from __future__ import annotations

from Skills.duplicate_toi_detector import (
    DuplicateDetectionResult,
    detect_duplicate_toi,
    format_duplicate_result,
)

_TOI_ROWS = [
    {"toi": "1234.01", "tic_id": 150428135, "period_days": 37.42, "epoch_bjd": 2458360.0},
    {"toi": "5678.01", "tic_id": 99999999, "period_days": 10.0, "epoch_bjd": 2458400.0},
]


class TestDetectDuplicateToi:
    def test_returns_result(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert isinstance(r, DuplicateDetectionResult)

    def test_exact_match_is_duplicate(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.flag == "DUPLICATE"
        assert r.is_duplicate

    def test_different_tic_not_duplicate(self) -> None:
        r = detect_duplicate_toi(11111111, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.flag == "UNIQUE"
        assert not r.is_duplicate

    def test_empty_toi_rows_no_data(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=[])
        assert r.flag == "NO_DATA"

    def test_period_mismatch_not_duplicate(self) -> None:
        r = detect_duplicate_toi(
            150428135, 100.0, 2458360.0, toi_rows=_TOI_ROWS,
            period_tol_days=0.01,
        )
        assert r.flag in {"UNIQUE", "POSSIBLE_DUPLICATE"}

    def test_best_match_populated(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.best_match is not None
        assert r.best_match.toi == "1234.01"

    def test_no_match_no_best_match(self) -> None:
        r = detect_duplicate_toi(11111111, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.best_match is None

    def test_input_values_preserved(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.input_tic_id == 150428135
        assert r.input_period == 37.42

    def test_flag_values_valid(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.flag in {"DUPLICATE", "POSSIBLE_DUPLICATE", "UNIQUE", "NO_DATA"}


class TestFormatDuplicateResult:
    def test_returns_string(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert isinstance(format_duplicate_result(r), str)

    def test_contains_tic_id(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert "150428135" in format_duplicate_result(r)

    def test_contains_flag(self) -> None:
        r = detect_duplicate_toi(150428135, 37.42, 2458360.0, toi_rows=_TOI_ROWS)
        assert r.flag in format_duplicate_result(r)
