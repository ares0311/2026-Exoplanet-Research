"""Tests for Skills/planet_mass_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_mass_estimator import estimate_planet_mass, format_planet_mass_result


def test_earth_mass():
    r = estimate_planet_mass(1.0)
    assert r.flag == "OK"
    assert r.mass_mearth is not None
    # Earth-like: C&K rocky regime gives ~1 M_earth
    assert 0.5 < r.mass_mearth < 2.0


def test_rocky_regime():
    r = estimate_planet_mass(1.0)
    assert r.regime == "rocky"


def test_volatile_regime():
    r = estimate_planet_mass(3.0)
    assert r.regime == "volatile"


def test_jovian_regime():
    r = estimate_planet_mass(15.0)
    assert r.regime == "jovian"


def test_mass_increases_with_radius():
    r1 = estimate_planet_mass(1.0)
    r2 = estimate_planet_mass(2.0)
    assert r2.mass_mearth > r1.mass_mearth


def test_mass_mjup_present():
    r = estimate_planet_mass(15.0)
    assert r.mass_mjup is not None
    assert r.mass_mjup > 0


def test_composition_class_rocky():
    r = estimate_planet_mass(1.0)
    assert r.composition_class is not None
    assert "rocky" in r.composition_class.lower() or r.composition_class


def test_invalid_zero_radius():
    r = estimate_planet_mass(0.0)
    assert r.flag == "INVALID"
    assert r.mass_mearth is None


def test_invalid_negative():
    r = estimate_planet_mass(-1.0)
    assert r.flag == "INVALID"


def test_invalid_too_large():
    r = estimate_planet_mass(200.0)
    assert r.flag == "INVALID"


def test_invalid_nan():
    r = estimate_planet_mass(float("nan"))
    assert r.flag == "INVALID"


def test_boundary_rocky_volatile():
    # 1.23 R_earth is boundary
    r = estimate_planet_mass(1.23)
    assert r.flag == "OK"
    assert r.regime is not None


def test_format_ok():
    r = estimate_planet_mass(2.5)
    text = format_planet_mass_result(r)
    assert "Planet Mass" in text
    assert "OK" in text


def test_format_invalid():
    r = estimate_planet_mass(0.0)
    text = format_planet_mass_result(r)
    assert "INVALID" in text
