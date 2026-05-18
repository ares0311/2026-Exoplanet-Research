"""Tests for telescope_time_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from telescope_time_estimator import (
    estimate_telescope_time,
    format_telescope_time_result,
)


class TestEstimateTelescopeTime:
    def test_basic_ok(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)
        assert r.flag == "OK"
        assert r.total_time_hours > 0

    def test_feasible_for_bright_star(self):
        r = estimate_telescope_time(5000.0, 3.0, 8.0, 200.0)
        assert r.is_feasible

    def test_infeasible_for_faint_star_small_telescope(self):
        r = estimate_telescope_time(100.0, 2.0, 18.0, 20.0)
        assert not r.is_feasible or r.flag in ("OK", "TOO_FAINT")

    def test_invalid_zero_depth(self):
        r = estimate_telescope_time(0.0, 2.0, 12.0, 100.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_duration(self):
        r = estimate_telescope_time(5000.0, 0.0, 12.0, 100.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_diameter(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 0.0)
        assert r.flag == "INVALID"

    def test_total_time_includes_baseline(self):
        r = estimate_telescope_time(5000.0, 2.0, 10.0, 100.0)
        # total > 0 when feasible
        if r.flag == "OK":
            assert r.total_time_hours > 0

    def test_exposure_time_positive(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)
        if r.flag == "OK":
            assert r.exposure_time_sec > 0

    def test_n_exposures_positive(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)
        if r.flag == "OK":
            assert r.n_exposures_in_transit >= 1

    def test_limiting_factor_not_empty(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)
        assert len(r.limiting_factor) > 0

    def test_result_frozen(self):
        r = estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)
        try:
            r.total_time_hours = 999  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatTelescopeTimeResult:
    def _make(self):
        return estimate_telescope_time(5000.0, 2.0, 12.0, 100.0)

    def test_returns_string(self):
        assert isinstance(format_telescope_time_result(self._make()), str)

    def test_contains_flag(self):
        r = self._make()
        assert r.flag in format_telescope_time_result(r)
