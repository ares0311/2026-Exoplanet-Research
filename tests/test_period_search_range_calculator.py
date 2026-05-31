"""Tests for Skills/period_search_range_calculator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_search_range_calculator import compute_search_range, format_search_range


class TestPeriodSearchRangeCalculator:
    def test_basic_ok(self) -> None:
        r = compute_search_range(27.0, 2.0)
        assert r.flag == "OK"

    def test_period_min_less_than_period_max(self) -> None:
        r = compute_search_range(27.0, 2.0)
        assert r.period_min_days < r.period_max_days

    def test_n_periods_positive(self) -> None:
        r = compute_search_range(27.0, 2.0)
        assert r.n_periods > 0

    def test_frequency_resolution_positive(self) -> None:
        r = compute_search_range(27.0, 2.0)
        assert r.frequency_resolution_per_day > 0.0

    def test_longer_baseline_more_periods(self) -> None:
        r1 = compute_search_range(27.0, 2.0)
        r2 = compute_search_range(270.0, 2.0)
        assert r2.n_periods > r1.n_periods

    def test_longer_baseline_wider_max(self) -> None:
        r1 = compute_search_range(27.0, 2.0)
        r2 = compute_search_range(270.0, 2.0)
        assert r2.period_max_days > r1.period_max_days

    def test_faster_cadence_shorter_min(self) -> None:
        r1 = compute_search_range(27.0, 30.0)
        r2 = compute_search_range(27.0, 2.0)
        assert r2.period_min_days <= r1.period_min_days

    def test_invalid_baseline(self) -> None:
        r = compute_search_range(0.0, 2.0)
        assert r.flag == "INVALID_BASELINE"
        assert r.n_periods == 0

    def test_invalid_cadence(self) -> None:
        r = compute_search_range(27.0, 0.0)
        assert r.flag == "INVALID_CADENCE"

    def test_min_period_at_least_half_day(self) -> None:
        r = compute_search_range(27.0, 1.0)
        assert r.period_min_days >= 0.5

    def test_freq_resolution_is_inverse_baseline(self) -> None:
        r = compute_search_range(27.0, 2.0)
        assert abs(r.frequency_resolution_per_day - 1.0 / 27.0) < 1e-4

    def test_format_returns_string(self) -> None:
        r = compute_search_range(27.0, 2.0)
        s = format_search_range(r)
        assert isinstance(s, str)
        assert "Period" in s
