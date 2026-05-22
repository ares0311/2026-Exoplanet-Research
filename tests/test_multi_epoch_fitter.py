"""Tests for multi_epoch_fitter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from multi_epoch_fitter import (
    fit_linear_ephemeris,
    format_ephemeris_fit_result,
)


def _perfect_transits(t0=2458000.0, period=3.5, n=10):
    nums = list(range(n))
    mids = [t0 + k * period for k in nums]
    return nums, mids


class TestFitLinearEphemeris:
    def test_invalid_empty(self):
        r = fit_linear_ephemeris([], [])
        assert r.flag == "INVALID"

    def test_invalid_mismatched(self):
        r = fit_linear_ephemeris([0, 1], [2458000.0])
        assert r.flag == "INVALID"

    def test_insufficient_two_points(self):
        r = fit_linear_ephemeris([0, 1], [2458000.0, 2458003.5])
        assert r.flag in ("INSUFFICIENT", "OK")

    def test_perfect_recovery(self):
        nums, mids = _perfect_transits(t0=2458000.0, period=3.5, n=8)
        r = fit_linear_ephemeris(nums, mids)
        assert r.flag == "OK"
        assert abs(r.period_days - 3.5) < 1e-4
        assert abs(r.t0 - 2458000.0) < 1e-4

    def test_rms_oc_near_zero_perfect(self):
        nums, mids = _perfect_transits(n=8)
        r = fit_linear_ephemeris(nums, mids)
        assert r.rms_oc_minutes < 0.01

    def test_n_transits_correct(self):
        nums, mids = _perfect_transits(n=6)
        r = fit_linear_ephemeris(nums, mids)
        assert r.n_transits == 6

    def test_oc_residuals_length(self):
        nums, mids = _perfect_transits(n=5)
        r = fit_linear_ephemeris(nums, mids)
        assert len(r.oc_residuals) == 5

    def test_chi2_reduced_present_n_gt_2(self):
        nums, mids = _perfect_transits(n=5)
        r = fit_linear_ephemeris(nums, mids)
        if r.flag == "OK":
            assert r.chi2_reduced is not None

    def test_with_midpoint_errors(self):
        nums, mids = _perfect_transits(n=8)
        errs = [0.001] * 8
        r = fit_linear_ephemeris(nums, mids, midpoint_errors=errs)
        assert r.flag == "OK"

    def test_period_err_positive(self):
        nums, mids = _perfect_transits(n=8)
        errs = [0.001] * 8
        r = fit_linear_ephemeris(nums, mids, midpoint_errors=errs)
        assert r.period_err is not None
        assert r.period_err >= 0

    def test_result_frozen(self):
        nums, mids = _perfect_transits(n=5)
        r = fit_linear_ephemeris(nums, mids)
        try:
            r.t0 = 0.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatEphemerisFitResult:
    def test_returns_string(self):
        nums, mids = _perfect_transits(n=5)
        r = fit_linear_ephemeris(nums, mids)
        assert isinstance(format_ephemeris_fit_result(r), str)

    def test_contains_flag(self):
        nums, mids = _perfect_transits(n=5)
        r = fit_linear_ephemeris(nums, mids)
        assert r.flag in format_ephemeris_fit_result(r)
