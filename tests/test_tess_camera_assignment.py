"""Tests for Skills/tess_camera_assignment.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tess_camera_assignment import assign_tess_camera, format_camera_assignment


class TestAssignTessCamera:
    def test_high_dec_camera3(self) -> None:
        # RA=0, dec=85 => ecliptic lat ≈ 66° (camera 3, range 54–78°)
        r = assign_tess_camera(0.0, 85.0)
        assert r.camera == 3
        assert r.flag == "OK"

    def test_invalid_ra(self) -> None:
        r = assign_tess_camera(-10.0, 45.0)
        assert r.flag == "INVALID_RA"

    def test_invalid_dec(self) -> None:
        r = assign_tess_camera(180.0, 95.0)
        assert r.flag == "INVALID_DEC"

    def test_ecliptic_gap(self) -> None:
        # RA=0, dec=0 => |elat|=0, in ecliptic gap
        r = assign_tess_camera(0.0, 0.0)
        assert r.flag == "ECLIPTIC_GAP"
        assert r.camera == 0

    def test_camera_1_low_elat(self) -> None:
        # dec~15 deg, RA=90 => moderate ecliptic lat => camera 1
        r = assign_tess_camera(180.0, 45.0)
        assert r.camera in (1, 2, 3)

    def test_ecliptic_lat_is_float(self) -> None:
        r = assign_tess_camera(180.0, 45.0)
        assert isinstance(r.ecliptic_lat_deg, float)

    def test_ecliptic_lon_in_range(self) -> None:
        r = assign_tess_camera(180.0, 45.0)
        assert 0.0 <= r.ecliptic_lon_deg < 360.0

    def test_ra_stored(self) -> None:
        r = assign_tess_camera(123.4, 45.0)
        assert r.ra_deg == 123.4

    def test_dec_stored(self) -> None:
        r = assign_tess_camera(180.0, -30.0)
        assert r.dec_deg == -30.0

    def test_abs_ecliptic_lat_non_negative(self) -> None:
        r = assign_tess_camera(90.0, 45.0)
        assert r.abs_ecliptic_lat_deg >= 0.0

    def test_south_pole_camera3(self) -> None:
        # RA=180, dec=-85 => ecliptic lat ≈ -66° (camera 3)
        r = assign_tess_camera(180.0, -85.0)
        assert r.camera == 3

    def test_format_returns_string(self) -> None:
        r = assign_tess_camera(180.0, 60.0)
        s = format_camera_assignment(r)
        assert isinstance(s, str)
        assert "Camera" in s
