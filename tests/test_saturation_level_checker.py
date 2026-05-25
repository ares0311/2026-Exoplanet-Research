"""Tests for Skills/saturation_level_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from saturation_level_checker import (
    check_saturation,
    format_saturation_result,
)


class TestCheckSaturation:
    def test_bright_star_saturated(self):
        result = check_saturation(5.0, saturation_threshold_adu=150_000.0)
        assert result.is_saturated is True
        assert result.flag == "OK"

    def test_faint_star_not_saturated(self):
        result = check_saturation(14.0, saturation_threshold_adu=150_000.0)
        assert result.is_saturated is False
        assert result.flag == "OK"

    def test_tmag_stored(self):
        result = check_saturation(12.0)
        assert result.tmag == 12.0

    def test_flux_ratio_positive(self):
        result = check_saturation(12.0)
        assert result.flux_ratio > 0.0

    def test_peak_flux_adu_positive(self):
        result = check_saturation(12.0)
        assert result.peak_flux_adu > 0.0

    def test_saturation_threshold_stored(self):
        result = check_saturation(12.0, saturation_threshold_adu=200_000.0)
        assert result.saturation_threshold_adu == 200_000.0

    def test_saturation_fraction_in_range(self):
        result = check_saturation(12.0)
        assert result.saturation_fraction >= 0.0

    def test_invalid_tmag_nan(self):
        result = check_saturation(float("nan"))
        assert result.flag == "INVALID"

    def test_invalid_tmag_inf(self):
        result = check_saturation(float("inf"))
        assert result.flag == "INVALID"

    def test_invalid_threshold_zero(self):
        result = check_saturation(12.0, saturation_threshold_adu=0.0)
        assert result.flag == "INVALID"

    def test_invalid_exposure_zero(self):
        result = check_saturation(12.0, exposure_sec=0.0)
        assert result.flag == "INVALID"

    def test_brighter_has_higher_flux_ratio(self):
        r_bright = check_saturation(8.0)
        r_faint = check_saturation(12.0)
        assert r_bright.flux_ratio > r_faint.flux_ratio

    def test_result_frozen(self):
        result = check_saturation(12.0)
        try:
            result.is_saturated = True
            raise AssertionError()
        except Exception:
            pass

    def test_format_returns_string(self):
        result = check_saturation(12.0)
        text = format_saturation_result(result)
        assert isinstance(text, str)

    def test_format_contains_tmag(self):
        result = check_saturation(12.0)
        text = format_saturation_result(result)
        assert "12" in text
