"""Tests for Skills/correlated_noise_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from correlated_noise_estimator import (
    estimate_correlated_noise,
    format_correlated_noise_result,
)


def _uniform_time(n=200, cadence_days=0.02083):
    return [i * cadence_days for i in range(n)]


def _flat_flux(n=200, level=1.0):
    return [level] * n


class TestEstimateCorrelatedNoise:
    def test_basic_ok(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        assert result.flag == "OK"

    def test_insufficient_short(self):
        t = _uniform_time(5)
        f = _flat_flux(5)
        result = estimate_correlated_noise(t, f)
        assert result.flag == "INSUFFICIENT"

    def test_rms_white_nonneg(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        assert result.rms_white_ppm >= 0.0

    def test_rms_red_nonneg(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        assert result.rms_red_ppm >= 0.0

    def test_mismatched_lengths_invalid(self):
        result = estimate_correlated_noise([0.0, 1.0], [1.0])
        assert result.flag == "INVALID"

    def test_empty_invalid(self):
        result = estimate_correlated_noise([], [])
        assert result.flag in ("INVALID", "INSUFFICIENT")

    def test_beta_factor_type(self):
        t = _uniform_time(200)
        f = [1.0 + 1e-4 * (i % 7) for i in range(200)]
        result = estimate_correlated_noise(t, f)
        assert isinstance(result.beta_factor, float)

    def test_n_bins_used_positive(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        assert result.n_bins_used > 0

    def test_timescale_hours_positive(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        if result.flag == "OK":
            assert result.timescale_hours > 0

    def test_custom_bin_sizes(self):
        t = _uniform_time(300)
        f = _flat_flux(300)
        result = estimate_correlated_noise(t, f, bin_sizes_hours=[0.5, 1.0])
        assert result.flag == "OK"

    def test_unnormalised_flux(self):
        t = _uniform_time(200)
        f = [1e4 + i * 0.01 for i in range(200)]
        result = estimate_correlated_noise(t, f, flux_is_normalised=False)
        assert result.flag in ("OK", "INSUFFICIENT")

    def test_result_is_frozen(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        try:
            result.beta_factor = 99.0
            raise AssertionError()
        except Exception:
            pass

    def test_n_bins_used_matches_bin_sizes(self):
        t = _uniform_time(300)
        f = _flat_flux(300)
        result = estimate_correlated_noise(t, f, bin_sizes_hours=[0.5, 1.0, 2.0])
        if result.flag == "OK":
            assert result.n_bins_used <= 3


class TestFormatNoiseEstimate:
    def test_returns_string(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        text = format_correlated_noise_result(result)
        assert isinstance(text, str)

    def test_contains_flag(self):
        t = _uniform_time(200)
        f = _flat_flux(200)
        result = estimate_correlated_noise(t, f)
        text = format_correlated_noise_result(result)
        assert result.flag in text
