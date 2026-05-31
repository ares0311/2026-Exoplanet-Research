"""Tests for Skills/sky_coordinate_validator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sky_coordinate_validator import format_coordinate_validation, validate_sky_coordinates


class TestSkyCoordinateValidator:
    def test_valid_coordinates(self) -> None:
        r = validate_sky_coordinates(180.0, 45.0)
        assert r.flag == "OK"

    def test_ecliptic_lat_computed(self) -> None:
        r = validate_sky_coordinates(180.0, 0.0)
        assert math.isfinite(r.ecliptic_lat_deg)

    def test_galactic_lat_computed(self) -> None:
        r = validate_sky_coordinates(180.0, 45.0)
        assert math.isfinite(r.galactic_lat_deg)

    def test_south_pole_cvz(self) -> None:
        # TESS CVZ near ecliptic south pole
        r = validate_sky_coordinates(90.0, -66.56)  # ecliptic pole region
        assert math.isfinite(r.ecliptic_lat_deg)

    def test_invalid_ra_negative(self) -> None:
        r = validate_sky_coordinates(-1.0, 0.0)
        assert r.flag == "RA_OUT_OF_RANGE"

    def test_invalid_ra_360(self) -> None:
        r = validate_sky_coordinates(360.0, 0.0)
        assert r.flag == "RA_OUT_OF_RANGE"

    def test_invalid_dec_too_high(self) -> None:
        r = validate_sky_coordinates(0.0, 91.0)
        assert r.flag == "DEC_OUT_OF_RANGE"

    def test_invalid_dec_too_low(self) -> None:
        r = validate_sky_coordinates(0.0, -91.0)
        assert r.flag == "DEC_OUT_OF_RANGE"

    def test_invalid_ra_nan(self) -> None:
        r = validate_sky_coordinates(float("nan"), 0.0)
        assert r.flag == "INVALID_RA"

    def test_invalid_dec_nan(self) -> None:
        r = validate_sky_coordinates(180.0, float("nan"))
        assert r.flag == "INVALID_DEC"

    def test_cvz_flag_near_ecliptic_pole(self) -> None:
        # Ecliptic north pole ≈ RA=270, Dec=66.56 (ecliptic lat +90)
        r = validate_sky_coordinates(270.0, 66.56)
        assert r.in_continuous_viewing_zone is True

    def test_format_returns_string(self) -> None:
        r = validate_sky_coordinates(180.0, 45.0)
        s = format_coordinate_validation(r)
        assert isinstance(s, str)
        assert "RA" in s
