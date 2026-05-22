"""Tests for Skills/significance_threshold_calculator.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from significance_threshold_calculator import (
    ThresholdResult,
    compute_bls_threshold,
    compute_snr_threshold,
    format_threshold_result,
)


def _flat_flux(n: int = 50) -> list[float]:
    return [1.0] * n


class TestComputeSnrThreshold:
    def test_returns_threshold_result(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        assert isinstance(result, ThresholdResult)

    def test_flag_ok_for_sufficient_data(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        assert result.flag == "OK"

    def test_too_few_points_returns_insufficient(self):
        result = compute_snr_threshold(_flat_flux(5))
        assert result.flag == "INSUFFICIENT"

    def test_empty_input_returns_insufficient(self):
        result = compute_snr_threshold([])
        assert result.flag == "INSUFFICIENT"

    def test_snr_threshold_is_non_negative(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        assert result.snr_threshold is not None
        assert result.snr_threshold >= 0.0

    def test_power_threshold_is_none(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        assert result.power_threshold is None

    def test_false_alarm_rate_stored(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, false_alarm_rate=0.05)
        assert result.false_alarm_rate == pytest.approx(0.05)

    def test_invalid_false_alarm_rate_returns_invalid(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, false_alarm_rate=0.0)
        assert result.flag == "INVALID"

    def test_invalid_false_alarm_rate_one_returns_invalid(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, false_alarm_rate=1.0)
        assert result.flag == "INVALID"

    def test_sigma_level_is_positive(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, false_alarm_rate=0.01)
        assert result.sigma_level is not None
        assert result.sigma_level > 0.0

    def test_n_bootstrap_stored(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, n_bootstrap=100)
        assert result.n_bootstrap == 100

    def test_too_few_bootstrap_returns_invalid(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux, n_bootstrap=5)
        assert result.flag == "INVALID"

    def test_frozen_dataclass(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestComputeBlsThreshold:
    def test_returns_threshold_result(self):
        flux = _flat_flux(50)
        result = compute_bls_threshold(flux, period_days=5.0)
        assert isinstance(result, ThresholdResult)

    def test_flag_ok_for_valid_input(self):
        flux = _flat_flux(50)
        result = compute_bls_threshold(flux, period_days=5.0)
        assert result.flag == "OK"

    def test_power_threshold_computed(self):
        flux = _flat_flux(50)
        result = compute_bls_threshold(flux, period_days=5.0)
        assert result.power_threshold is not None
        assert result.power_threshold >= 0.0

    def test_snr_threshold_also_computed(self):
        flux = _flat_flux(50)
        result = compute_bls_threshold(flux, period_days=5.0)
        assert result.snr_threshold is not None

    def test_too_few_points_returns_insufficient(self):
        result = compute_bls_threshold(_flat_flux(5), period_days=5.0)
        assert result.flag == "INSUFFICIENT"

    def test_negative_period_returns_insufficient(self):
        flux = _flat_flux(50)
        result = compute_bls_threshold(flux, period_days=-1.0)
        assert result.flag == "INSUFFICIENT"


class TestFormatThresholdResult:
    def test_returns_string(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        md = format_threshold_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        flux = _flat_flux(50)
        result = compute_snr_threshold(flux)
        md = format_threshold_result(result)
        assert result.flag in md

    def test_insufficient_format(self):
        result = compute_snr_threshold([])
        md = format_threshold_result(result)
        assert "INSUFFICIENT" in md
