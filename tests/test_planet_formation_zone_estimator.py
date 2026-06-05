"""Tests for Skills/planet_formation_zone_estimator.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_formation_zone_estimator import (
    FormationZonesResult,
    estimate_formation_zones,
    format_formation_zones_result,
)


class TestEstimateFormationZones:
    def test_returns_result_type(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert isinstance(r, FormationZonesResult)

    def test_flag_ok(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.flag == "OK"

    def test_solar_snow_line_near_2_7(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.snow_line_au == pytest.approx(2.7, rel=0.01)

    def test_silicate_line_inside_snow_line(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.silicate_line_au < r.snow_line_au

    def test_co2_line_outside_snow_line(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.co2_line_au > r.snow_line_au

    def test_co_line_outside_co2_line(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.co_line_au > r.co2_line_au

    def test_inner_hole_inside_silicate_line(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.inner_hole_au < r.silicate_line_au

    def test_hz_inner_less_than_hz_outer(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.habitable_zone_inner_au < r.habitable_zone_outer_au

    def test_hz_inner_near_1_au_for_solar(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0, stellar_teff_k=5778.0)
        # Earth is in the HZ
        assert r.habitable_zone_inner_au < 1.0 < r.habitable_zone_outer_au

    def test_brighter_star_larger_zones(self):
        r1 = estimate_formation_zones(stellar_luminosity_lsun=4.0)
        r2 = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r1.snow_line_au > r2.snow_line_au

    def test_scales_as_sqrt_luminosity(self):
        r1 = estimate_formation_zones(stellar_luminosity_lsun=4.0)
        r2 = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r1.snow_line_au == pytest.approx(2.0 * r2.snow_line_au, rel=0.01)

    def test_invalid_luminosity(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=0.0)
        assert r.flag != "OK"
        assert math.isnan(r.snow_line_au)

    def test_negative_luminosity(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=-1.0)
        assert r.flag != "OK"

    def test_rocky_zone_outer_equals_snow_line(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        assert r.rocky_zone_outer_au == pytest.approx(r.snow_line_au, rel=0.01)

    def test_frozen_dataclass(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        try:
            r.snow_line_au = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatFormationZonesResult:
    def test_ok_returns_table(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=1.0)
        out = format_formation_zones_result(r)
        assert "snow line" in out.lower() or "Snow" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = estimate_formation_zones(stellar_luminosity_lsun=0.0)
        out = format_formation_zones_result(r)
        assert "flag=" in out
