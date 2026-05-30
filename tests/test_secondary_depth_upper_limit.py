"""Tests for Skills/secondary_depth_upper_limit.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from secondary_depth_upper_limit import (
    SecondaryLimitResult,
    compute_secondary_upper_limit,
    format_secondary_limit,
)


class TestComputeSecondaryUpperLimit:
    def test_basic_calculation(self):
        flux = [1.0, 1.001, 0.999, 1.002, 0.998, 1.001, 0.999]
        result = compute_secondary_upper_limit(flux)
        assert isinstance(result, SecondaryLimitResult)
        assert result.upper_limit_ppm >= 0.0

    def test_n_sigma_1_gives_smaller_limit(self):
        flux = [1.0, 1.001, 0.999, 1.002, 0.998, 1.001, 0.999]
        result_3 = compute_secondary_upper_limit(flux, n_sigma=3.0)
        result_1 = compute_secondary_upper_limit(flux, n_sigma=1.0)
        assert result_1.upper_limit_ppm < result_3.upper_limit_ppm

    def test_constant_flux_zero_noise(self):
        flux = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = compute_secondary_upper_limit(flux)
        assert result.noise_ppm == 0.0
        assert result.upper_limit_ppm == 0.0

    def test_empty_list_no_data(self):
        result = compute_secondary_upper_limit([])
        assert result.flag == "NO_DATA"

    def test_insufficient_data_two_points(self):
        result = compute_secondary_upper_limit([1.0, 1.001])
        assert result.flag == "INSUFFICIENT_DATA"

    def test_noise_ppm_positive_for_noisy_data(self):
        flux = [1.0, 1.01, 0.99, 1.005, 0.995, 1.003, 0.997]
        result = compute_secondary_upper_limit(flux)
        assert result.noise_ppm > 0.0

    def test_upper_limit_equals_n_sigma_times_noise(self):
        flux = [1.0, 1.002, 0.998, 1.001, 0.999, 1.003, 0.997]
        result = compute_secondary_upper_limit(flux, n_sigma=2.5)
        expected = result.n_sigma * result.noise_ppm
        assert abs(result.upper_limit_ppm - expected) < 1e-9

    def test_format_returns_string(self):
        flux = [1.0, 1.001, 0.999, 1.002, 0.998]
        result = compute_secondary_upper_limit(flux)
        md = format_secondary_limit(result)
        assert isinstance(md, str)

    def test_flag_ok_for_sufficient_data(self):
        flux = [1.0, 1.001, 0.999, 1.002, 0.998, 1.001, 0.999]
        result = compute_secondary_upper_limit(flux)
        assert result.flag == "OK"

    def test_upper_limit_positive_for_noisy_data(self):
        flux = [1.0, 1.01, 0.99, 1.005, 0.995]
        result = compute_secondary_upper_limit(flux)
        assert result.upper_limit_ppm > 0.0

    def test_mad_formula_five_element(self):
        # flux = [0.999, 1.000, 1.001, 1.002, 1.003]
        # median = 1.001; deviations = [0.002, 0.001, 0.000, 0.001, 0.002]
        # MAD = median([0.002, 0.001, 0.000, 0.001, 0.002]) = 0.001
        # noise_ppm = 0.001 * 1.4826 * 1e6 = 1482.6
        flux = [0.999, 1.000, 1.001, 1.002, 1.003]
        result = compute_secondary_upper_limit(flux)
        assert abs(result.noise_ppm - 1482.6) < 1.0

    def test_format_contains_ppm(self):
        flux = [1.0, 1.001, 0.999, 1.002, 0.998]
        result = compute_secondary_upper_limit(flux)
        md = format_secondary_limit(result)
        assert "ppm" in md
