"""Tests for Skills/stellar_surface_gravity_estimator.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_surface_gravity_estimator import estimate_surface_gravity, format_surface_gravity_result


def test_solar_logg():
    # Solar logg ≈ 4.44
    r = estimate_surface_gravity(1.0, 1.0)
    assert r.flag == "OK"
    assert r.logg is not None
    assert abs(r.logg - 4.44) < 0.05


def test_gravity_cgs():
    r = estimate_surface_gravity(1.0, 1.0)
    assert r.gravity_cgs is not None
    assert 2e4 < r.gravity_cgs < 3e4  # ≈ 27400 cm/s²


def test_larger_radius_lower_gravity():
    r1 = estimate_surface_gravity(1.0, 1.0)
    r2 = estimate_surface_gravity(1.0, 2.0)
    assert r2.logg is not None and r1.logg is not None
    assert r2.logg < r1.logg


def test_larger_mass_higher_gravity():
    r1 = estimate_surface_gravity(1.0, 1.0)
    r2 = estimate_surface_gravity(2.0, 1.0)
    assert r2.logg is not None and r1.logg is not None
    assert r2.logg > r1.logg


def test_invalid_zero_mass():
    r = estimate_surface_gravity(0.0, 1.0)
    assert r.flag == "INVALID"
    assert r.logg is None


def test_invalid_zero_radius():
    r = estimate_surface_gravity(1.0, 0.0)
    assert r.flag == "INVALID"


def test_invalid_negative():
    r = estimate_surface_gravity(-1.0, 1.0)
    assert r.flag == "INVALID"


def test_invalid_nan():
    r = estimate_surface_gravity(float("nan"), 1.0)
    assert r.flag == "INVALID"


def test_uncertainty_propagation():
    r = estimate_surface_gravity(1.0, 1.0, mass_err_msun=0.05, radius_err_rsun=0.02)
    assert r.flag == "OK"
    assert r.logg_uncertainty is not None
    assert r.logg_uncertainty > 0


def test_uncertainty_none_without_errors():
    r = estimate_surface_gravity(1.0, 1.0)
    assert r.logg_uncertainty is None


def test_logg_relation():
    # logg = log10(G * M / R^2) in CGS; 4×mass → logg += log10(4) ≈ 0.602
    r1 = estimate_surface_gravity(1.0, 1.0)
    r4 = estimate_surface_gravity(4.0, 1.0)
    assert r4.logg is not None and r1.logg is not None
    assert abs(r4.logg - r1.logg - math.log10(4.0)) < 0.01


def test_format_ok():
    r = estimate_surface_gravity(1.0, 1.0)
    text = format_surface_gravity_result(r)
    assert "Surface Gravity" in text
    assert "OK" in text


def test_format_invalid():
    r = estimate_surface_gravity(0.0, 1.0)
    text = format_surface_gravity_result(r)
    assert "INVALID" in text


def test_result_fields():
    r = estimate_surface_gravity(1.5, 1.3)
    assert r.mass_msun == 1.5
    assert r.radius_rsun == 1.3
