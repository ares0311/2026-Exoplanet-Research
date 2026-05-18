"""Tests for chi_square_period_checker.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from chi_square_period_checker import (
    check_chi_square_period,
    format_chi_square_period_result,
)


def _make_sinusoidal(n=200, period=5.0, depth=0.01):
    """Phase-folded sinusoidal light curve with clear period."""
    time = [i * 0.1 for i in range(n)]
    flux = [1.0 - depth * (math.sin(2 * math.pi * t / period) ** 2) for t in time]
    return time, flux


class TestCheckChiSquarePeriod:
    def test_significant_period(self):
        time, flux = _make_sinusoidal(200, period=5.0, depth=0.02)
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        assert r.flag in ("SIGNIFICANT", "MARGINAL")

    def test_flat_lc_not_significant(self):
        time = [float(i) for i in range(100)]
        flux = [1.0] * 100
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        # Flat LC has chi2_folded=0 → INSUFFICIENT, or NOT_SIGNIFICANT
        assert r.flag in ("NOT_SIGNIFICANT", "INSUFFICIENT")

    def test_chi2_null_positive(self):
        time, flux = _make_sinusoidal()
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        assert r.chi2_null >= 0

    def test_chi2_folded_leq_null(self):
        time, flux = _make_sinusoidal(200, period=5.0, depth=0.05)
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        # Folded model fits better → chi2_folded <= chi2_null
        if r.flag != "INSUFFICIENT":
            assert r.chi2_folded <= r.chi2_null + 1e-6

    def test_p_value_in_range(self):
        time, flux = _make_sinusoidal()
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        assert 0.0 <= r.p_value <= 1.0

    def test_insufficient_too_few_points(self):
        r = check_chi_square_period([1.0, 2.0], [1.0, 1.0], 5.0, 0.0)
        assert r.flag == "INSUFFICIENT"

    def test_invalid_period_zero(self):
        time = list(range(50))
        flux = [1.0] * 50
        r = check_chi_square_period(time, flux, 0.0, 0.0)
        assert r.flag == "INSUFFICIENT"

    def test_custom_flux_err(self):
        time, flux = _make_sinusoidal(100)
        errs = [0.001] * 100
        r = check_chi_square_period(time, flux, 5.0, 0.0, flux_err=errs)
        assert r.flag in ("SIGNIFICANT", "MARGINAL", "NOT_SIGNIFICANT", "INSUFFICIENT")

    def test_f_statistic_nonnegative(self):
        time, flux = _make_sinusoidal(200)
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        assert r.f_statistic >= 0.0

    def test_is_significant_consistent(self):
        time, flux = _make_sinusoidal(200, depth=0.05)
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        if r.flag == "SIGNIFICANT":
            assert r.is_significant
        elif r.flag == "NOT_SIGNIFICANT":
            assert not r.is_significant

    def test_result_frozen(self):
        time, flux = _make_sinusoidal(50)
        r = check_chi_square_period(time, flux, 5.0, 0.0)
        try:
            r.p_value = 0.5  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatChiSquarePeriodResult:
    def _make(self):
        time, flux = _make_sinusoidal(200)
        return check_chi_square_period(time, flux, 5.0, 0.0)

    def test_returns_string(self):
        assert isinstance(format_chi_square_period_result(self._make()), str)

    def test_contains_flag(self):
        r = self._make()
        assert r.flag in format_chi_square_period_result(r)
