"""Tests for Skills/planetary_magnetic_moment_estimator.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planetary_magnetic_moment_estimator import (
    MagneticMomentResult,
    estimate_magnetic_moment,
    format_magnetic_moment_result,
)


class TestEstimateMagneticMoment:
    def test_returns_result_type(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert isinstance(r, MagneticMomentResult)

    def test_flag_ok(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert r.flag == "OK"

    def test_earth_like_is_near_one(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0,
                                      rotation_period_days=1.0, core_fraction=0.3)
        assert r.magnetic_moment_earth_units == pytest.approx(1.0, rel=0.01)

    def test_surface_field_positive(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert r.surface_field_gauss > 0.0

    def test_magnetospheric_radius_positive(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert r.magnetospheric_radius_rp > 0.0

    def test_dynamo_class_string(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert r.dynamo_class in ("STRONG", "MODERATE", "WEAK", "ABSENT")

    def test_faster_rotation_stronger_field(self):
        r1 = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0,
                                       rotation_period_days=0.5)
        r2 = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0,
                                       rotation_period_days=5.0)
        assert r1.magnetic_moment_earth_units > r2.magnetic_moment_earth_units

    def test_more_massive_planet_stronger_field(self):
        r1 = estimate_magnetic_moment(planet_mass_mearth=10.0, planet_radius_rearth=2.0)
        r2 = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        assert r1.magnetic_moment_earth_units > r2.magnetic_moment_earth_units

    def test_invalid_mass(self):
        r = estimate_magnetic_moment(planet_mass_mearth=0.0, planet_radius_rearth=1.0)
        assert r.flag != "OK"
        assert math.isnan(r.magnetic_moment_earth_units)

    def test_invalid_radius(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=0.0)
        assert r.flag != "OK"

    def test_invalid_period(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0,
                                      rotation_period_days=0.0)
        assert r.flag != "OK"

    def test_large_planet_strong_dynamo(self):
        r = estimate_magnetic_moment(planet_mass_mearth=300.0, planet_radius_rearth=10.0,
                                      rotation_period_days=0.4)
        assert r.dynamo_class in ("STRONG", "MODERATE")

    def test_frozen_dataclass(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        try:
            r.magnetic_moment_earth_units = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatMagneticMomentResult:
    def test_ok_returns_table(self):
        r = estimate_magnetic_moment(planet_mass_mearth=1.0, planet_radius_rearth=1.0)
        out = format_magnetic_moment_result(r)
        assert "Magnetic moment" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = estimate_magnetic_moment(planet_mass_mearth=0.0, planet_radius_rearth=1.0)
        out = format_magnetic_moment_result(r)
        assert "flag=" in out
