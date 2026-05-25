"""Tests for Skills/stellar_luminosity_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_luminosity_calculator import compute_stellar_luminosity, format_luminosity_result


def test_solar_values():
    r = compute_stellar_luminosity(5778.0, 1.0)
    assert r.flag == "OK"
    assert r.luminosity_lsun is not None
    assert abs(r.luminosity_lsun - 1.0) < 0.01


def test_luminosity_scales_with_radius():
    r1 = compute_stellar_luminosity(5778.0, 1.0)
    r2 = compute_stellar_luminosity(5778.0, 2.0)
    assert r2.luminosity_lsun is not None and r1.luminosity_lsun is not None
    assert abs(r2.luminosity_lsun / r1.luminosity_lsun - 4.0) < 0.01


def test_luminosity_scales_with_teff():
    r1 = compute_stellar_luminosity(5778.0, 1.0)
    r2 = compute_stellar_luminosity(5778.0 * 2, 1.0)
    assert r2.luminosity_lsun / r1.luminosity_lsun == pytest_approx(16.0, rel=0.01)


def pytest_approx(val, rel=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - val) / abs(val) < rel
    return _Approx()


def test_log10_luminosity():
    r = compute_stellar_luminosity(5778.0, 1.0)
    assert r.luminosity_log10 is not None
    assert abs(r.luminosity_log10 - 0.0) < 0.01


def test_radius_au():
    r = compute_stellar_luminosity(5778.0, 1.0)
    assert r.radius_au is not None
    assert abs(r.radius_au - 0.00465047) < 1e-5


def test_invalid_zero_teff():
    r = compute_stellar_luminosity(0.0, 1.0)
    assert r.flag == "INVALID"
    assert r.luminosity_lsun is None


def test_invalid_zero_radius():
    r = compute_stellar_luminosity(5778.0, 0.0)
    assert r.flag == "INVALID"


def test_invalid_negative_radius():
    r = compute_stellar_luminosity(5778.0, -1.0)
    assert r.flag == "INVALID"


def test_invalid_nan():
    r = compute_stellar_luminosity(float("nan"), 1.0)
    assert r.flag == "INVALID"


def test_invalid_inf():
    r = compute_stellar_luminosity(5778.0, float("inf"))
    assert r.flag == "INVALID"


def test_cool_star():
    r = compute_stellar_luminosity(3500.0, 0.3)
    assert r.flag == "OK"
    assert r.luminosity_lsun < 0.1


def test_hot_star():
    r = compute_stellar_luminosity(30000.0, 5.0)
    assert r.flag == "OK"
    assert r.luminosity_lsun > 1000.0


def test_format_contains_luminosity():
    r = compute_stellar_luminosity(5778.0, 1.0)
    text = format_luminosity_result(r)
    assert "Stellar Luminosity" in text
    assert "OK" in text


def test_format_invalid():
    r = compute_stellar_luminosity(0.0, 1.0)
    text = format_luminosity_result(r)
    assert "INVALID" in text


def test_result_fields():
    r = compute_stellar_luminosity(5778.0, 1.0)
    assert r.teff_k == 5778.0
    assert r.radius_rsun == 1.0
