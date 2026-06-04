"""Tests for Skills/photometric_contamination_flag.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from photometric_contamination_flag import (
    compute_photometric_contamination_flag,
    format_contamination_flag_result,
)


class TestComputePhotometricContaminationFlag:
    def test_ok_flag(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 15.0)
        assert r.flag == "OK"

    def test_large_delta_mag_negligible(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 20.0)
        assert r.contamination_level == "NEGLIGIBLE"

    def test_equal_brightness_severe(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 10.0)
        assert r.contamination_level in ("SEVERE", "EXTREME")

    def test_flux_ratio_positive(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 12.0)
        assert r.flux_ratio > 0.0

    def test_dilution_factor_between_zero_and_one(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 12.0)
        assert 0.0 < r.dilution_factor < 1.0

    def test_depth_correction_greater_than_one(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 12.0)
        assert r.depth_correction_factor > 1.0

    def test_separation_reduces_contamination(self) -> None:
        r_near = compute_photometric_contamination_flag(
            10.0, 12.0, separation_arcsec=5.0, aperture_radius_arcsec=21.0
        )
        r_far = compute_photometric_contamination_flag(
            10.0, 12.0, separation_arcsec=30.0, aperture_radius_arcsec=21.0
        )
        assert r_near.dilution_factor > r_far.dilution_factor

    def test_depth_bias_ppm_positive(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 13.0)
        assert r.depth_bias_ppm_per_1000ppm > 0.0

    def test_invalid_aperture(self) -> None:
        r = compute_photometric_contamination_flag(
            10.0, 12.0, separation_arcsec=5.0, aperture_radius_arcsec=0.0
        )
        assert r.flag == "INVALID_APERTURE"

    def test_delta_mag_computed_correctly(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 13.0)
        assert abs(r.delta_mag - 3.0) < 1e-10

    def test_result_frozen(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 15.0)
        try:
            r.dilution_factor = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_photometric_contamination_flag(10.0, 15.0)
        s = format_contamination_flag_result(r)
        assert isinstance(s, str)
        assert r.flag in s
