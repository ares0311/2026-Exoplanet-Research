"""Tests for Skills/period_search_grid_optimizer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from period_search_grid_optimizer import compute_period_search_grid, format_period_grid_result


class TestComputePeriodSearchGrid:
    def test_ok_flag(self) -> None:
        r = compute_period_search_grid(27.0)
        assert r.flag == "OK"

    def test_n_periods_positive(self) -> None:
        r = compute_period_search_grid(27.0, period_min_days=0.5, period_max_days=13.0)
        assert r.n_periods > 0

    def test_longer_baseline_more_periods(self) -> None:
        r_short = compute_period_search_grid(27.0)
        r_long = compute_period_search_grid(365.0)
        assert r_long.n_periods > r_short.n_periods

    def test_frequency_resolution_correct(self) -> None:
        import math
        r = compute_period_search_grid(100.0)
        expected = 1.0 / (math.pi * 100.0)
        assert abs(r.frequency_resolution_per_day - expected) < 1e-9

    def test_period_spacing_at_min_smaller_than_max(self) -> None:
        r = compute_period_search_grid(27.0, period_min_days=0.5, period_max_days=13.0)
        assert r.period_spacing_at_min < r.period_spacing_at_max

    def test_oversampling_increases_n_periods(self) -> None:
        r1 = compute_period_search_grid(27.0, oversampling=1.0)
        r2 = compute_period_search_grid(27.0, oversampling=2.0)
        assert r2.n_periods > r1.n_periods

    def test_wider_period_range_more_periods(self) -> None:
        r_narrow = compute_period_search_grid(27.0, period_min_days=1.0, period_max_days=5.0)
        r_wide = compute_period_search_grid(27.0, period_min_days=0.5, period_max_days=13.0)
        assert r_wide.n_periods > r_narrow.n_periods

    def test_invalid_baseline(self) -> None:
        r = compute_period_search_grid(0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_invalid_period_min(self) -> None:
        r = compute_period_search_grid(27.0, period_min_days=0.0)
        assert r.flag == "INVALID_PERIOD_MIN"

    def test_invalid_period_max_less_than_min(self) -> None:
        r = compute_period_search_grid(27.0, period_min_days=5.0, period_max_days=3.0)
        assert r.flag == "INVALID_PERIOD_MAX"

    def test_result_frozen(self) -> None:
        r = compute_period_search_grid(27.0)
        try:
            r.n_periods = 0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_period_search_grid(27.0)
        s = format_period_grid_result(r)
        assert isinstance(s, str)
        assert r.flag in s
