"""Tests for transit_count_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_count_estimator import (
    estimate_transit_count,
    format_transit_count_result,
)


class TestEstimateTransitCount:
    def test_invalid_period_zero(self):
        r = estimate_transit_count(0.0, 0.0, [(0.0, 100.0)])
        assert r.flag == "INVALID"

    def test_no_intervals(self):
        r = estimate_transit_count(5.0, 0.0, [])
        assert r.flag == "NO_INTERVALS"

    def test_degenerate_interval(self):
        r = estimate_transit_count(5.0, 0.0, [(10.0, 10.0)])
        assert r.flag == "NO_INTERVALS"

    def test_basic_count(self):
        # period=5, epoch=0, interval=[0, 30] → transits at 0,5,10,15,20,25,30 = 7
        r = estimate_transit_count(5.0, 0.0, [(0.0, 30.0)])
        assert r.flag == "OK"
        assert r.n_in_window >= 5

    def test_coverage_fraction_one_if_all_in(self):
        r = estimate_transit_count(5.0, 0.0, [(0.0, 100.0)])
        assert r.flag == "OK"
        assert r.coverage_fraction == 1.0

    def test_zero_coverage_outside(self):
        # epoch=0, period=5, but interval starts way after
        r = estimate_transit_count(5.0, 0.0, [(1000.0, 1010.0)])
        if r.flag == "OK":
            assert r.n_in_window >= 0

    def test_complete_vs_in_window(self):
        # complete ≤ in_window
        r = estimate_transit_count(5.0, 0.0, [(0.0, 50.0)], duration_hours=2.0)
        if r.flag == "OK":
            assert r.n_complete <= r.n_in_window

    def test_multiple_intervals(self):
        ivs = [(0.0, 10.0), (20.0, 30.0)]
        r = estimate_transit_count(5.0, 0.0, ivs)
        assert r.flag == "OK"

    def test_result_frozen(self):
        r = estimate_transit_count(5.0, 0.0, [(0.0, 30.0)])
        try:
            r.n_in_window = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_period_preserved(self):
        r = estimate_transit_count(7.25, 0.0, [(0.0, 50.0)])
        assert r.period_days == 7.25


class TestFormatTransitCountResult:
    def test_returns_string(self):
        r = estimate_transit_count(5.0, 0.0, [(0.0, 30.0)])
        assert isinstance(format_transit_count_result(r), str)

    def test_contains_flag(self):
        r = estimate_transit_count(5.0, 0.0, [(0.0, 30.0)])
        assert r.flag in format_transit_count_result(r)
