"""Tests for Skills/direct_imaging_contrast_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from direct_imaging_contrast_estimator import (
    compute_direct_imaging_contrast,
    format_direct_imaging_result,
)


class TestComputeDirectImagingContrast:
    def test_ok_flag_valid_inputs(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        assert r.flag == "OK"

    def test_angular_separation_correct(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        assert abs(r.angular_separation_arcsec - 0.5) < 1e-4

    def test_contrast_positive(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        assert r.contrast_ratio > 0

    def test_wider_orbit_larger_separation(self) -> None:
        r1 = compute_direct_imaging_contrast(1.0, 10.0, 11.0)
        r2 = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        assert r2.angular_separation_arcsec > r1.angular_separation_arcsec

    def test_nearer_star_larger_separation(self) -> None:
        r_near = compute_direct_imaging_contrast(5.0, 5.0, 11.0)
        r_far = compute_direct_imaging_contrast(5.0, 20.0, 11.0)
        assert r_near.angular_separation_arcsec > r_far.angular_separation_arcsec

    def test_larger_albedo_higher_contrast(self) -> None:
        r1 = compute_direct_imaging_contrast(5.0, 10.0, 11.0, geometric_albedo=0.1)
        r2 = compute_direct_imaging_contrast(5.0, 10.0, 11.0, geometric_albedo=0.5)
        assert r2.contrast_ratio > r1.contrast_ratio

    def test_mag_diff_positive(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        assert r.contrast_mag_diff > 0

    def test_roman_detectable_direct_imaging_scenario(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 100.0, geometric_albedo=0.5)
        assert r.angular_separation_arcsec >= 0.1

    def test_invalid_orbital_distance(self) -> None:
        r = compute_direct_imaging_contrast(0.0, 10.0, 11.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_invalid_stellar_distance(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 0.0, 11.0)
        assert r.flag == "INVALID_STELLAR_DISTANCE"

    def test_invalid_radius(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_result_frozen(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        try:
            r.contrast_ratio = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_direct_imaging_contrast(5.0, 10.0, 11.0)
        s = format_direct_imaging_result(r)
        assert isinstance(s, str)
        assert r.flag in s
