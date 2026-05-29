"""Tests for Skills/atmospheric_dispersion_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from atmospheric_dispersion_estimator import (  # noqa: E402
    DispersionResult,
    estimate_atmospheric_dispersion,
    format_dispersion_result,
)


def test_returns_dataclass():
    r = estimate_atmospheric_dispersion(400.0, 700.0)
    assert isinstance(r, DispersionResult)


def test_dispersion_positive():
    r = estimate_atmospheric_dispersion(400.0, 700.0)
    assert r.dispersion_arcsec > 0


def test_same_wavelength_zero_dispersion():
    r = estimate_atmospheric_dispersion(500.0, 500.0)
    assert r.dispersion_arcsec == 0.0


def test_order_invariant():
    r1 = estimate_atmospheric_dispersion(400.0, 700.0)
    r2 = estimate_atmospheric_dispersion(700.0, 400.0)
    assert abs(r1.dispersion_arcsec - r2.dispersion_arcsec) < 1e-6


def test_higher_airmass_more_dispersion():
    r1 = estimate_atmospheric_dispersion(400.0, 700.0, airmass=1.2)
    r2 = estimate_atmospheric_dispersion(400.0, 700.0, airmass=2.5)
    assert r2.dispersion_arcsec > r1.dispersion_arcsec


def test_large_dispersion_flag():
    # At high airmass and wide wavelength range, dispersion > 1 arcsec
    r = estimate_atmospheric_dispersion(350.0, 800.0, airmass=3.0)
    assert r.flag == "LARGE_DISPERSION"


def test_ok_flag_narrow_range():
    r = estimate_atmospheric_dispersion(580.0, 620.0, airmass=1.2)
    assert r.flag == "OK"


def test_airmass_stored():
    r = estimate_atmospheric_dispersion(400.0, 700.0, airmass=1.8)
    assert abs(r.airmass - 1.8) < 0.01


def test_wavelengths_stored():
    r = estimate_atmospheric_dispersion(450.0, 750.0, airmass=1.5)
    assert r.wavelength1_nm == 450.0
    assert r.wavelength2_nm == 750.0


def test_zenith_angle_range():
    r = estimate_atmospheric_dispersion(400.0, 700.0, airmass=1.5)
    assert 0.0 <= r.zenith_angle_deg < 90.0


def test_format_returns_string():
    r = estimate_atmospheric_dispersion(400.0, 700.0)
    s = format_dispersion_result(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = estimate_atmospheric_dispersion(400.0, 700.0)
    s = format_dispersion_result(r)
    assert "Flag" in s
