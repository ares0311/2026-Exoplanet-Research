"""Tests for Skills/target_coordinates_converter.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from target_coordinates_converter import convert_coordinates, format_coordinate_result


def test_basic_conversion():
    r = convert_coordinates(0.0, 0.0)
    assert r.flag == "OK"
    assert r.ra_deg == 0.0
    assert r.dec_deg == 0.0


def test_ecliptic_coords_present():
    r = convert_coordinates(90.0, 0.0)
    assert r.ecl_lon_deg is not None
    assert r.ecl_lat_deg is not None


def test_galactic_coords_present():
    r = convert_coordinates(90.0, 0.0)
    assert r.gal_lon_deg is not None
    assert r.gal_lat_deg is not None


def test_ecliptic_vernal_equinox():
    # At RA=0, Dec=0, ecliptic longitude should be 0
    r = convert_coordinates(0.0, 0.0)
    assert r.ecl_lon_deg is not None
    assert abs(r.ecl_lon_deg) < 1.0 or abs(r.ecl_lon_deg - 360.0) < 1.0


def test_ecliptic_lat_range():
    r = convert_coordinates(180.0, 45.0)
    assert r.ecl_lat_deg is not None
    assert -90.0 <= r.ecl_lat_deg <= 90.0


def test_galactic_lat_range():
    r = convert_coordinates(180.0, 45.0)
    assert r.gal_lat_deg is not None
    assert -90.0 <= r.gal_lat_deg <= 90.0


def test_galactic_lon_range():
    r = convert_coordinates(180.0, 45.0)
    assert r.gal_lon_deg is not None
    assert 0.0 <= r.gal_lon_deg < 360.0


def test_invalid_dec_too_large():
    r = convert_coordinates(0.0, 91.0)
    assert r.flag == "INVALID"


def test_invalid_dec_too_small():
    r = convert_coordinates(0.0, -91.0)
    assert r.flag == "INVALID"


def test_invalid_nan_ra():
    r = convert_coordinates(float("nan"), 0.0)
    assert r.flag == "INVALID"


def test_invalid_nan_dec():
    r = convert_coordinates(0.0, float("nan"))
    assert r.flag == "INVALID"


def test_ra_wraps():
    r = convert_coordinates(360.0, 0.0)
    assert r.flag == "OK"


def test_galactic_center_approx():
    # Galactic centre at RA≈266.4°, Dec≈-28.9° → gal_lon≈0, gal_lat≈0
    r = convert_coordinates(266.4, -28.9)
    assert r.gal_lon_deg is not None
    assert abs(r.gal_lon_deg) < 5.0 or abs(r.gal_lon_deg - 360.0) < 5.0
    assert abs(r.gal_lat_deg) < 5.0


def test_format_ok():
    r = convert_coordinates(90.0, 30.0)
    text = format_coordinate_result(r)
    assert "Coordinate" in text
    assert "OK" in text


def test_format_invalid():
    r = convert_coordinates(0.0, 95.0)
    text = format_coordinate_result(r)
    assert "INVALID" in text
