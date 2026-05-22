"""Tests for Skills/polynomial_detrend.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from polynomial_detrend import (
    DetrenderResult,
    apply_detrend,
    fit_polynomial_trend,
    format_detrend_result,
)


class TestFitPolynomialTrend:
    def test_returns_detrender_result(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux)
        assert isinstance(result, DetrenderResult)

    def test_flag_ok_for_valid_input(self):
        time = [float(i) for i in range(20)]
        flux = [1.0 + 0.01 * i for i in range(20)]
        result = fit_polynomial_trend(time, flux)
        assert result.flag == "OK"

    def test_empty_time_returns_invalid(self):
        result = fit_polynomial_trend([], [])
        assert result.flag == "INVALID"

    def test_single_point_returns_invalid(self):
        result = fit_polynomial_trend([0.0], [1.0])
        assert result.flag == "INVALID"

    def test_mismatched_lengths_returns_invalid(self):
        result = fit_polynomial_trend([0.0, 1.0], [1.0])
        assert result.flag == "INVALID"

    def test_negative_degree_returns_invalid(self):
        time = [float(i) for i in range(10)]
        flux = [1.0] * 10
        result = fit_polynomial_trend(time, flux, degree=-1)
        assert result.flag == "INVALID"

    def test_zero_segments_returns_invalid(self):
        time = [float(i) for i in range(10)]
        flux = [1.0] * 10
        result = fit_polynomial_trend(time, flux, n_segments=0)
        assert result.flag == "INVALID"

    def test_detrended_flux_length_matches_input(self):
        time = [float(i) for i in range(20)]
        flux = [1.0 + 0.001 * i for i in range(20)]
        result = fit_polynomial_trend(time, flux, degree=1)
        assert len(result.detrended_flux) == 20

    def test_degree_stored_correctly(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux, degree=3)
        assert result.degree == 3

    def test_n_segments_stored_correctly(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux, n_segments=2)
        assert result.n_segments == 2

    def test_rms_before_and_after_computed(self):
        time = [float(i) for i in range(20)]
        flux = [1.0 + 0.01 * i for i in range(20)]
        result = fit_polynomial_trend(time, flux, degree=1)
        assert result.rms_before is not None
        assert result.rms_after is not None

    def test_mask_excludes_in_transit_points(self):
        time = [float(i) for i in range(30)]
        flux = [1.0] * 30
        flux[10] = 0.99  # simulated transit
        mask = [True] * 30
        mask[10] = False
        result = fit_polynomial_trend(time, flux, degree=1, mask=mask)
        assert result.flag == "OK"

    def test_constant_flux_rms_after_near_zero(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux, degree=0)
        assert result.rms_after is not None
        assert result.rms_after < 1e-6

    def test_frozen_dataclass(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestApplyDetrend:
    def test_returns_list(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux)
        out = apply_detrend(time, flux, result)
        assert isinstance(out, list)

    def test_length_preserved(self):
        time = [float(i) for i in range(20)]
        flux = [1.0 + 0.01 * i for i in range(20)]
        result = fit_polynomial_trend(time, flux, degree=1)
        out = apply_detrend(time, flux, result)
        assert len(out) == 20


class TestFormatDetrenderResult:
    def test_returns_string(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux)
        md = format_detrend_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        time = [float(i) for i in range(20)]
        flux = [1.0] * 20
        result = fit_polynomial_trend(time, flux)
        md = format_detrend_result(result)
        assert "OK" in md

    def test_invalid_result_format(self):
        result = fit_polynomial_trend([], [])
        md = format_detrend_result(result)
        assert "INVALID" in md
