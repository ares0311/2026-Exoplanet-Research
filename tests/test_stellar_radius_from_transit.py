"""Tests for Skills/stellar_radius_from_transit.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_radius_from_transit import derive_stellar_radius, format_radius_result


class TestStellarRadiusFromTransit:
    def test_sun_like_star(self) -> None:
        # Earth transit depth ~84 ppm; planet_radius~1 Rearth → R* ~1 Rsun
        r = derive_stellar_radius(84.0, 1.0)
        assert r.flag == "OK"
        assert 0.8 < r.stellar_radius_rsun < 1.3

    def test_radius_ratio_correct(self) -> None:
        # depth = (Rp/R*)^2 = 0.01 → Rp/R* = 0.1
        r = derive_stellar_radius(10000.0, 10.0)  # 1% depth
        assert abs(r.radius_ratio - 0.1) < 1e-4

    def test_invalid_depth_zero(self) -> None:
        r = derive_stellar_radius(0.0, 1.0)
        assert r.flag == "INVALID_DEPTH"
        assert math.isnan(r.stellar_radius_rsun)

    def test_invalid_depth_negative(self) -> None:
        r = derive_stellar_radius(-100.0, 1.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_planet_radius(self) -> None:
        r = derive_stellar_radius(1000.0, 0.0)
        assert r.flag == "INVALID_PLANET_RADIUS"

    def test_error_propagation_with_depth_err(self) -> None:
        r = derive_stellar_radius(1000.0, 10.0, depth_err_ppm=50.0)
        assert r.flag == "OK"
        assert r.stellar_radius_rsun_err > 0.0

    def test_default_systematic_error(self) -> None:
        r = derive_stellar_radius(1000.0, 10.0)
        assert r.flag == "OK"
        # Default 5% error
        assert abs(r.stellar_radius_rsun_err / r.stellar_radius_rsun - 0.05) < 1e-3

    def test_implausible_small_star(self) -> None:
        # Very deep transit with tiny planet → implausibly small star
        r = derive_stellar_radius(900000.0, 0.1)
        assert r.flag == "IMPLAUSIBLE_STELLAR_RADIUS"

    def test_radius_ratio_stored(self) -> None:
        r = derive_stellar_radius(1000.0, 10.0)
        assert r.flag == "OK"
        assert r.radius_ratio > 0.0

    def test_larger_planet_larger_star(self) -> None:
        r1 = derive_stellar_radius(1000.0, 10.0)
        r2 = derive_stellar_radius(1000.0, 20.0)
        assert r2.stellar_radius_rsun > r1.stellar_radius_rsun

    def test_deeper_transit_smaller_star(self) -> None:
        r1 = derive_stellar_radius(1000.0, 10.0)
        r2 = derive_stellar_radius(4000.0, 10.0)
        assert r2.stellar_radius_rsun < r1.stellar_radius_rsun

    def test_format_returns_string(self) -> None:
        r = derive_stellar_radius(1000.0, 10.0)
        s = format_radius_result(r)
        assert isinstance(s, str)
        assert "Stellar radius" in s
