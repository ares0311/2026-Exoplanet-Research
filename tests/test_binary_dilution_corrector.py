"""Tests for Skills/binary_dilution_corrector.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from binary_dilution_corrector import (
    BinaryDilutionResult,
    correct_for_binary_dilution,
    format_binary_dilution_result,
)


class TestCorrectForBinaryDilution:
    def test_returns_result_type(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert isinstance(r, BinaryDilutionResult)

    def test_flag_ok(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert r.flag == "OK"

    def test_corrected_depth_larger_than_observed(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert r.corrected_depth_ppm > 5000.0

    def test_dilution_factor_between_0_and_1(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert 0.0 < r.dilution_factor < 1.0

    def test_corrected_rp_rs_positive(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert r.corrected_rp_rs > 0.0

    def test_zero_delta_mag_equal_flux(self):
        # Same magnitude companion → equal brightness → dilution ~ 0.5
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=0.0)
        assert r.dilution_factor == pytest.approx(0.5, abs=0.01)
        assert r.corrected_depth_ppm == pytest.approx(10000.0, rel=0.01)

    def test_larger_delta_mag_less_dilution(self):
        r1 = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=5.0)
        r2 = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=1.0)
        assert r1.dilution_factor < r2.dilution_factor

    def test_flux_ratio_override(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0,
                                         flux_ratio_override=0.5)
        assert r.flag == "OK"
        assert r.dilution_factor == pytest.approx(0.5 / 1.5, rel=0.01)

    def test_fpp_inflation_factor_gte_1(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert r.fpp_inflation_factor >= 1.0

    def test_invalid_depth(self):
        r = correct_for_binary_dilution(observed_depth_ppm=0.0, delta_magnitude=2.0)
        assert r.flag != "OK"
        assert math.isnan(r.corrected_depth_ppm)

    def test_invalid_flux_ratio_override(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0,
                                         flux_ratio_override=-0.1)
        assert r.flag != "OK"

    def test_depth_ratio_finite(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        assert math.isfinite(r.depth_ratio)

    def test_frozen_dataclass(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        try:
            r.corrected_depth_ppm = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatBinaryDilutionResult:
    def test_ok_returns_table(self):
        r = correct_for_binary_dilution(observed_depth_ppm=5000.0, delta_magnitude=2.0)
        out = format_binary_dilution_result(r)
        assert "Dilution factor" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = correct_for_binary_dilution(observed_depth_ppm=0.0, delta_magnitude=2.0)
        out = format_binary_dilution_result(r)
        assert "flag=" in out
