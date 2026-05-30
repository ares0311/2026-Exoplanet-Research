"""Tests for Skills/mission_overlap_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from mission_overlap_checker import check_mission_overlap, format_mission_overlap


class TestCheckMissionOverlap:
    def test_tess_always_included(self) -> None:
        r = check_mission_overlap(180.0, 45.0)
        assert "TESS" in r.missions

    def test_kepler_field(self) -> None:
        r = check_mission_overlap(295.0, 44.0)
        assert r.in_kepler_field is True
        assert "Kepler" in r.missions

    def test_outside_kepler_field(self) -> None:
        r = check_mission_overlap(180.0, 0.0)
        assert r.in_kepler_field is False

    def test_k2_near_ecliptic(self) -> None:
        # vernal equinox RA=0, dec=0 lies exactly on the ecliptic plane
        r = check_mission_overlap(0.0, 0.0)
        assert r.in_k2_accessible is True

    def test_invalid_ra(self) -> None:
        r = check_mission_overlap(-10.0, 45.0)
        assert r.flag == "INVALID_RA"

    def test_invalid_dec(self) -> None:
        r = check_mission_overlap(180.0, 95.0)
        assert r.flag == "INVALID_DEC"

    def test_tess_only_flag(self) -> None:
        r = check_mission_overlap(180.0, 45.0)
        assert r.flag in ("TESS_ONLY", "MULTI_MISSION")

    def test_multi_mission_flag(self) -> None:
        r = check_mission_overlap(295.0, 44.0)
        assert r.flag == "MULTI_MISSION"

    def test_ecliptic_lat_is_float(self) -> None:
        r = check_mission_overlap(180.0, 45.0)
        assert isinstance(r.ecliptic_lat_deg, float)

    def test_ra_stored(self) -> None:
        r = check_mission_overlap(123.4, 45.0)
        assert r.ra_deg == 123.4

    def test_dec_stored(self) -> None:
        r = check_mission_overlap(180.0, -30.0)
        assert r.dec_deg == -30.0

    def test_format_returns_string(self) -> None:
        r = check_mission_overlap(180.0, 45.0)
        s = format_mission_overlap(r)
        assert isinstance(s, str)
        assert "Mission" in s
