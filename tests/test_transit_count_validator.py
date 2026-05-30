"""Tests for Skills/transit_count_validator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_count_validator import format_transit_count_validation, validate_transit_count


class TestValidateTransitCount:
    def test_consistent_count(self) -> None:
        # period=5, baseline=30 → min_expected=6, max=7; use n=6 which is in range
        r = validate_transit_count(6, 5.0, 30.0)
        assert r.is_consistent is True
        assert r.flag == "OK"

    def test_inconsistent_too_many(self) -> None:
        r = validate_transit_count(100, 5.0, 30.0)
        assert r.is_consistent is False
        assert r.flag == "INCONSISTENT_COUNT"

    def test_zero_transits_consistent(self) -> None:
        # period=100, baseline=10 → min=0, max=1; n=0 is within range
        r = validate_transit_count(0, 100.0, 10.0)
        assert r.is_consistent is True

    def test_invalid_period(self) -> None:
        r = validate_transit_count(5, 0.0, 30.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_baseline(self) -> None:
        r = validate_transit_count(5, 5.0, 0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_invalid_gap_fraction(self) -> None:
        r = validate_transit_count(5, 5.0, 30.0, gap_fraction=1.5)
        assert r.flag == "INVALID_GAP_FRACTION"

    def test_gap_fraction_reduces_min(self) -> None:
        r_no_gap = validate_transit_count(3, 5.0, 30.0, gap_fraction=0.0)
        r_with_gap = validate_transit_count(3, 5.0, 30.0, gap_fraction=0.5)
        assert r_with_gap.expected_min <= r_no_gap.expected_min

    def test_expected_max_formula(self) -> None:
        r = validate_transit_count(5, 5.0, 30.0)
        assert r.expected_max >= 6

    def test_n_transits_stored(self) -> None:
        r = validate_transit_count(7, 5.0, 30.0)
        assert r.n_transits == 7

    def test_is_consistent_bool(self) -> None:
        r = validate_transit_count(5, 5.0, 30.0)
        assert isinstance(r.is_consistent, bool)

    def test_format_returns_string(self) -> None:
        r = validate_transit_count(5, 5.0, 30.0)
        s = format_transit_count_validation(r)
        assert isinstance(s, str)
        assert "Transit" in s

    def test_single_transit_consistent(self) -> None:
        r = validate_transit_count(1, 20.0, 25.0)
        assert r.is_consistent is True
