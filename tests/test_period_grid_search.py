"""Tests for period_grid_search.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from period_grid_search import (
    format_period_grid_result,
    search_period_grid,
)


def _flat_lc(n=100):
    time = [i * 0.02 for i in range(n)]
    flux = [1.0] * n
    return time, flux


def _transit_lc(n=200, period=5.0, depth=0.01):
    time = [i * 0.02 for i in range(n)]
    flux = []
    for t in time:
        phase = (t % period) / period
        if phase < 0.02 or phase > 0.98:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return time, flux


class TestSearchPeriodGrid:
    def test_result_frozen(self):
        t, f = _flat_lc()
        r = search_period_grid(t, f, [5.0, 10.0])
        try:
            r.n_periods_tested = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_empty_inputs_invalid(self):
        r = search_period_grid([], [], [])
        assert r.flag == "INVALID"

    def test_empty_period_grid_invalid(self):
        t, f = _flat_lc()
        r = search_period_grid(t, f, [])
        assert r.flag == "INVALID"

    def test_too_few_points_invalid(self):
        r = search_period_grid([1.0, 2.0], [1.0, 1.0], [5.0])
        assert r.flag == "INVALID"

    def test_n_periods_tested_matches_grid(self):
        t, f = _flat_lc()
        grid = [2.0, 3.0, 4.0, 5.0]
        r = search_period_grid(t, f, grid)
        assert r.n_periods_tested == len(grid)

    def test_peak_period_in_grid(self):
        t, f = _flat_lc()
        grid = [2.0, 3.0, 5.0]
        r = search_period_grid(t, f, grid)
        if r.best_period_days is not None:
            assert r.best_period_days in grid

    def test_transit_signal_detected(self):
        t, f = _transit_lc(n=300, period=5.0, depth=0.02)
        grid = [4.8, 5.0, 5.2, 10.0]
        r = search_period_grid(t, f, grid)
        assert r.flag in ("OK", "INSUFFICIENT")
        assert r.best_period_days is not None

    def test_powers_same_length_as_grid(self):
        t, f = _flat_lc(50)
        grid = [2.0, 3.0, 4.0]
        r = search_period_grid(t, f, grid)
        assert len(r.powers) == len(grid)

    def test_periods_tuple_matches_grid(self):
        t, f = _flat_lc(50)
        grid = [2.0, 3.0]
        r = search_period_grid(t, f, grid)
        assert r.periods_days == tuple(grid)

    def test_with_flux_err(self):
        t, f = _flat_lc(100)
        err = [0.001] * 100
        r = search_period_grid(t, f, [5.0, 10.0], flux_err=err)
        assert r.flag in ("OK", "INSUFFICIENT", "INVALID")

    def test_mismatched_flux_length_invalid(self):
        t = [0.1 * i for i in range(50)]
        f = [1.0] * 30
        r = search_period_grid(t, f, [5.0])
        assert r.flag == "INVALID"

    def test_format_returns_string(self):
        t, f = _flat_lc()
        r = search_period_grid(t, f, [5.0])
        s = format_period_grid_result(r)
        assert isinstance(s, str)
        assert "Period" in s

    def test_best_power_not_none_when_ok(self):
        t, f = _transit_lc(300, 5.0, 0.01)
        r = search_period_grid(t, f, [4.9, 5.0, 5.1])
        if r.flag == "OK":
            assert r.best_power is not None
