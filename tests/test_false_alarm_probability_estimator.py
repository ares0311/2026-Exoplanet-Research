"""Tests for false_alarm_probability_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from false_alarm_probability_estimator import (
    estimate_fap,
    format_fap_result,
)


class TestEstimateFAP:
    def test_analytic_basic(self):
        r = estimate_fap(10.0, 1000, 5.0)
        assert r.flag in ("SIGNIFICANT", "MARGINAL", "NOT_SIGNIFICANT")
        assert 0.0 <= r.fap <= 1.0

    def test_high_power_significant(self):
        r = estimate_fap(30.0, 5000, 3.0)
        assert r.flag == "SIGNIFICANT"
        assert r.fap < 0.01

    def test_low_power_not_significant(self):
        r = estimate_fap(0.5, 100, 5.0)
        assert r.flag == "NOT_SIGNIFICANT"

    def test_fap_in_range(self):
        r = estimate_fap(5.0, 500, 10.0)
        assert 0.0 <= r.fap <= 1.0

    def test_log10_fap_matches(self):
        import math
        r = estimate_fap(10.0, 1000, 5.0)
        if r.fap > 0:
            assert abs(r.log10_fap - math.log10(r.fap)) < 0.01

    def test_sigma_nonnegative(self):
        r = estimate_fap(10.0, 1000, 5.0)
        assert r.significance_sigma >= 0.0

    def test_empirical_method(self):
        noise = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        r = estimate_fap(5.5, 1000, 3.0, method="empirical", noise_powers=noise)
        assert r.method == "empirical"
        # 4 noise powers >= 5.5 → fap = 4/10
        assert abs(r.fap - 0.4) < 0.01

    def test_empirical_falls_back_to_analytic(self):
        r = estimate_fap(10.0, 1000, 5.0, method="empirical", noise_powers=None)
        assert r.method == "analytic"

    def test_invalid_zero_cadences(self):
        r = estimate_fap(10.0, 0, 5.0)
        assert r.fap == 1.0

    def test_invalid_negative_power(self):
        r = estimate_fap(-1.0, 1000, 5.0)
        assert r.fap == 1.0

    def test_n_eff_positive(self):
        r = estimate_fap(10.0, 1000, 5.0)
        assert r.n_independent_frequencies > 0

    def test_result_frozen(self):
        r = estimate_fap(10.0, 1000, 5.0)
        try:
            r.fap = 0.5  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatFAPResult:
    def _make(self):
        return estimate_fap(10.0, 1000, 5.0)

    def test_returns_string(self):
        assert isinstance(format_fap_result(self._make()), str)

    def test_contains_flag(self):
        r = self._make()
        assert r.flag in format_fap_result(r)
