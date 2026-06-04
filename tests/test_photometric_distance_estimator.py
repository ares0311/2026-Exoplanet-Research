"""Tests for Skills/photometric_distance_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from photometric_distance_estimator import (
    compute_photometric_distance,
    format_photometric_distance_result,
)


class TestComputePhotometricDistance:
    def test_ok_flag(self) -> None:
        r = compute_photometric_distance(5.03, 5778.0, 1.0)
        assert r.flag == "OK"

    def test_sun_at_10pc_correct_distance(self) -> None:
        # Sun at 10 pc: m = M_bol_sun + 5*log10(10) - 5 = 4.74 + 0 = 4.74
        r = compute_photometric_distance(4.74, 5778.0, 1.0)
        assert abs(r.distance_pc - 10.0) < 0.5

    def test_luminosity_solar_for_sun(self) -> None:
        r = compute_photometric_distance(4.74, 5778.0, 1.0)
        assert abs(r.luminosity_lsun - 1.0) < 0.05

    def test_distance_pc_positive(self) -> None:
        r = compute_photometric_distance(8.0, 5000.0, 0.8)
        assert r.distance_pc > 0.0

    def test_distance_ly_positive(self) -> None:
        r = compute_photometric_distance(8.0, 5000.0, 0.8)
        assert r.distance_ly > 0.0

    def test_fainter_star_farther_away(self) -> None:
        r_near = compute_photometric_distance(6.0, 5778.0, 1.0)
        r_far = compute_photometric_distance(10.0, 5778.0, 1.0)
        assert r_far.distance_pc > r_near.distance_pc

    def test_extinction_increases_distance(self) -> None:
        r_no_ext = compute_photometric_distance(8.0, 5778.0, 1.0, extinction_mag=0.0)
        r_ext = compute_photometric_distance(8.0, 5778.0, 1.0, extinction_mag=0.5)
        assert r_no_ext.distance_pc > r_ext.distance_pc

    def test_invalid_teff(self) -> None:
        r = compute_photometric_distance(5.0, 0.0, 1.0)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_radius(self) -> None:
        r = compute_photometric_distance(5.0, 5778.0, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_extinction(self) -> None:
        r = compute_photometric_distance(5.0, 5778.0, 1.0, extinction_mag=-0.1)
        assert r.flag == "INVALID_EXTINCTION"

    def test_result_frozen(self) -> None:
        r = compute_photometric_distance(5.0, 5778.0, 1.0)
        try:
            r.distance_pc = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_photometric_distance(5.0, 5778.0, 1.0)
        s = format_photometric_distance_result(r)
        assert isinstance(s, str)
        assert r.flag in s
