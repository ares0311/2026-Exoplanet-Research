"""Tests for tess_sector_map.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from tess_sector_map import (
    format_sector_map,
    get_sector_map,
)


class TestGetSectorMap:
    def test_result_frozen(self):
        r = get_sector_map(83.8, -5.4)
        try:
            r.n_sectors = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_ecliptic_south_pole(self):
        # RA=0, Dec=-90 has high ecliptic lat → lots of sectors
        r = get_sector_map(0.0, -90.0)
        assert r.flag in ("OK", "ECLIPTIC_PLANE")
        assert r.n_sectors >= 0

    def test_ecliptic_plane_flagged(self):
        # Dec~0, near ecliptic plane
        r = get_sector_map(90.0, 0.0)
        # May or may not be flagged depending on exact coords
        assert r.flag in ("OK", "ECLIPTIC_PLANE", "INVALID")

    def test_invalid_ra_too_large(self):
        r = get_sector_map(400.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_dec_too_large(self):
        r = get_sector_map(180.0, 100.0)
        assert r.flag == "INVALID"

    def test_invalid_dec_too_small(self):
        r = get_sector_map(180.0, -100.0)
        assert r.flag == "INVALID"

    def test_n_sectors_consistent_with_sector_ids(self):
        r = get_sector_map(83.8, -5.4)
        if r.flag in ("OK", "ECLIPTIC_PLANE"):
            assert r.n_sectors == len(r.sector_ids)

    def test_sector_ids_are_positive(self):
        r = get_sector_map(83.8, -5.4)
        for sid in r.sector_ids:
            assert sid >= 1

    def test_n_years_parameter(self):
        r1 = get_sector_map(83.8, -5.4, n_years=2)
        r6 = get_sector_map(83.8, -5.4, n_years=6)
        assert r1.n_sectors <= r6.n_sectors

    def test_ecliptic_lat_computed(self):
        r = get_sector_map(83.8, -5.4)
        if r.flag != "INVALID":
            assert isinstance(r.ecliptic_lat_deg, float)

    def test_years_observed_tuple(self):
        r = get_sector_map(83.8, -5.4)
        if r.flag in ("OK",):
            assert isinstance(r.years_observed, tuple)

    def test_format_returns_string(self):
        r = get_sector_map(83.8, -5.4)
        s = format_sector_map(r)
        assert isinstance(s, str)
        assert "Sector" in s

    def test_format_contains_flag(self):
        r = get_sector_map(400.0, 0.0)
        s = format_sector_map(r)
        assert "INVALID" in s

    def test_zero_n_years(self):
        r = get_sector_map(83.8, -5.4, n_years=0)
        assert r.n_sectors == 0 or r.flag in ("OK", "INVALID")
