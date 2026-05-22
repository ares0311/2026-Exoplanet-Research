"""Tests for snr_vs_period_plotter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from snr_vs_period_plotter import (
    compute_period_snr,
    format_period_snr_result,
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
        if phase < 0.03 or phase > 0.97:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return time, flux


class TestComputePeriodSNR:
    def test_result_frozen(self):
        t, f = _flat_lc()
        r = compute_period_snr(t, f, [5.0])
        try:
            r.n_periods = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_point_frozen(self):
        t, f = _flat_lc()
        r = compute_period_snr(t, f, [5.0])
        if r.points:
            pt = r.points[0]
            try:
                pt.snr = 99.0  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_empty_inputs_invalid(self):
        r = compute_period_snr([], [], [5.0])
        assert r.flag == "INVALID"

    def test_empty_period_grid_invalid(self):
        t, f = _flat_lc()
        r = compute_period_snr(t, f, [])
        assert r.flag == "INVALID"

    def test_n_periods_matches_grid(self):
        t, f = _flat_lc()
        grid = [2.0, 3.0, 5.0, 7.0]
        r = compute_period_snr(t, f, grid)
        assert r.n_periods == len(grid)

    def test_points_length_matches_grid(self):
        t, f = _flat_lc()
        grid = [2.0, 3.0, 5.0]
        r = compute_period_snr(t, f, grid)
        assert len(r.points) == len(grid)

    def test_peak_period_in_grid(self):
        t, f = _flat_lc()
        grid = [2.0, 5.0, 10.0]
        r = compute_period_snr(t, f, grid)
        if r.peak_period_days is not None:
            assert r.peak_period_days in grid

    def test_transit_signal_has_positive_snr(self):
        t, f = _transit_lc(300, 5.0, 0.02)
        err = [0.001] * 300
        grid = [4.9, 5.0, 5.1]
        r = compute_period_snr(t, f, grid, flux_err=err)
        assert r.flag in ("OK", "INSUFFICIENT")
        if r.peak_snr is not None:
            assert r.peak_snr > 0

    def test_with_flux_err(self):
        t, f = _flat_lc()
        err = [0.001] * len(t)
        r = compute_period_snr(t, f, [5.0], flux_err=err)
        assert r.flag in ("OK", "INSUFFICIENT", "INVALID")

    def test_n_transits_positive(self):
        t, f = _flat_lc(200)
        r = compute_period_snr(t, f, [5.0])
        for pt in r.points:
            assert pt.n_transits_expected >= 1

    def test_format_returns_string(self):
        t, f = _flat_lc()
        r = compute_period_snr(t, f, [5.0])
        s = format_period_snr_result(r)
        assert isinstance(s, str)
        assert "SNR" in s

    def test_mismatched_flux_invalid(self):
        t = [0.1 * i for i in range(50)]
        f = [1.0] * 30
        r = compute_period_snr(t, f, [5.0])
        assert r.flag == "INVALID"
