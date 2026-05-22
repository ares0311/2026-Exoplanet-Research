"""Tests for Skills/photometric_binary_checker.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photometric_binary_checker import (
    PhotometricBinaryResult,
    check_photometric_binary,
    format_binary_check_result,
)


def _flat_lc(n: int = 100, dt: float = 0.1) -> tuple[list[float], list[float]]:
    time = [i * dt for i in range(n)]
    flux = [1.0] * n
    return time, flux


def _ellipsoidal_lc(
    period: float = 5.0, n: int = 200, dt: float = 0.1, amplitude: float = 0.002
) -> tuple[list[float], list[float]]:
    time = [i * dt for i in range(n)]
    flux = [1.0 + amplitude * math.cos(4.0 * math.pi * t / period) for t in time]
    return time, flux


class TestCheckPhotometricBinary:
    def test_returns_photometric_binary_result(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        assert isinstance(result, PhotometricBinaryResult)

    def test_flag_ok_for_valid_input(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        assert result.flag == "OK"

    def test_empty_input_returns_invalid(self):
        result = check_photometric_binary([], [], 5.0, 0.0)
        assert result.flag == "INVALID"

    def test_too_few_points_returns_invalid(self):
        time, flux = _flat_lc(5)
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        assert result.flag == "INVALID"

    def test_negative_period_returns_invalid(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, -1.0, 0.0)
        assert result.flag == "INVALID"

    def test_flat_lc_not_binary_candidate(self):
        time, flux = _flat_lc(200)
        result = check_photometric_binary(time, flux, 5.0, 0.0, snr_threshold=3.0)
        assert not result.is_binary_candidate

    def test_ellipsoidal_lc_may_be_binary(self):
        time, flux = _ellipsoidal_lc(period=5.0, amplitude=0.01)
        result = check_photometric_binary(time, flux, 5.0, 0.0, snr_threshold=2.0)
        assert result.half_period_amplitude_ppm is not None

    def test_amplitudes_are_non_negative(self):
        time, flux = _flat_lc(100)
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        if result.half_period_amplitude_ppm is not None:
            assert result.half_period_amplitude_ppm >= 0.0
        if result.full_period_amplitude_ppm is not None:
            assert result.full_period_amplitude_ppm >= 0.0

    def test_with_flux_err(self):
        time, flux = _flat_lc()
        err = [0.001] * len(time)
        result = check_photometric_binary(time, flux, 5.0, 0.0, flux_err=err)
        assert result.flag == "OK"

    def test_mismatched_lengths_returns_invalid(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux[:-1], 5.0, 0.0)
        assert result.flag == "INVALID"

    def test_frozen_dataclass(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatBinaryCheckResult:
    def test_returns_string(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        md = format_binary_check_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        time, flux = _flat_lc()
        result = check_photometric_binary(time, flux, 5.0, 0.0)
        md = format_binary_check_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = check_photometric_binary([], [], 5.0, 0.0)
        md = format_binary_check_result(result)
        assert "INVALID" in md
