"""Tests for Skills/planet_insolation_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_insolation_calculator import InsolationResult, compute_insolation


class TestPlanetInsolationCalculator:
    def test_earth_like_insolation(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0, albedo=0.3)
        assert r.flag == "OK"
        assert abs(r.insolation_searth - 1.0) < 0.1

    def test_earth_like_teq(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0, albedo=0.3)
        assert 250 < r.equilibrium_temp_k < 300

    def test_invalid_luminosity_zero(self) -> None:
        r = compute_insolation(luminosity_lsun=0.0, semi_major_axis_au=1.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_invalid_luminosity_negative(self) -> None:
        r = compute_insolation(luminosity_lsun=-1.0, semi_major_axis_au=1.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_invalid_sma_zero(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=0.0)
        assert r.flag == "INVALID_SEMI_MAJOR_AXIS"

    def test_invalid_albedo_negative(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0, albedo=-0.1)
        assert r.flag == "INVALID_ALBEDO"

    def test_invalid_albedo_one(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0, albedo=1.0)
        assert r.flag == "INVALID_ALBEDO"

    def test_hz_classification_inner(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=0.5, albedo=0.3)
        assert r.flag == "OK"
        assert isinstance(r.in_hz, bool)

    def test_brighter_star_higher_insolation(self) -> None:
        r1 = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0)
        r2 = compute_insolation(luminosity_lsun=4.0, semi_major_axis_au=1.0)
        assert r2.insolation_searth > r1.insolation_searth

    def test_closer_orbit_higher_insolation(self) -> None:
        r1 = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0)
        r2 = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=0.5)
        assert r2.insolation_searth > r1.insolation_searth

    def test_result_is_frozen(self) -> None:
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0)
        assert isinstance(r, InsolationResult)
        try:
            object.__setattr__(r, "flag", "mutated")
            raise AssertionError()
        except Exception:
            pass

    def test_format_not_empty(self) -> None:
        from planet_insolation_calculator import format_insolation_result
        r = compute_insolation(luminosity_lsun=1.0, semi_major_axis_au=1.0)
        s = format_insolation_result(r)
        assert len(s) > 10
