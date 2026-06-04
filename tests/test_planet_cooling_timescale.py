"""Tests for Skills/planet_cooling_timescale.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_cooling_timescale import (
    compute_planet_cooling_timescale,
    format_planet_cooling_result,
)


class TestComputePlanetCoolingTimescale:
    def test_ok_flag_default(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        assert r.flag == "OK"

    def test_ok_flag_with_lint(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0, intrinsic_luminosity_lsun=1e-9)
        assert r.flag == "OK"

    def test_ok_flag_with_age(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0, planet_age_gyr=5.0)
        assert r.flag == "OK"

    def test_timescale_positive(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        assert r.kelvin_helmholtz_timescale_gyr > 0.0

    def test_grav_energy_positive(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        assert r.gravitational_energy_j > 0.0

    def test_intrinsic_luminosity_positive(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        assert r.intrinsic_luminosity_w > 0.0

    def test_more_massive_longer_timescale(self) -> None:
        r1 = compute_planet_cooling_timescale(1.0, 1.0)
        r2 = compute_planet_cooling_timescale(10.0, 1.0)
        assert r2.kelvin_helmholtz_timescale_gyr != r1.kelvin_helmholtz_timescale_gyr

    def test_cooling_class_set(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        assert r.cooling_class in ("YOUNG_INFLATED", "CONTRACTING", "EVOLVED")

    def test_high_lint_young_inflated(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0, intrinsic_luminosity_lsun=0.01)
        assert r.cooling_class == "YOUNG_INFLATED"

    def test_invalid_mass(self) -> None:
        r = compute_planet_cooling_timescale(0.0, 1.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_radius(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_luminosity(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0, intrinsic_luminosity_lsun=0.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_format_returns_string(self) -> None:
        r = compute_planet_cooling_timescale(1.0, 1.0)
        s = format_planet_cooling_result(r)
        assert isinstance(s, str)
        assert r.flag in s
