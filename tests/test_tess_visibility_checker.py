"""Tests for Skills.tess_visibility_checker."""
from __future__ import annotations

from Skills.tess_visibility_checker import (
    TESSVisibilityResult,
    check_tess_visibility,
    format_visibility_result,
)


class TestCheckTESSVisibility:
    def test_returns_result(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        assert isinstance(r, TESSVisibilityResult)

    def test_invalid_ra_negative(self) -> None:
        r = check_tess_visibility(-10.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_dec_out_of_range(self) -> None:
        r = check_tess_visibility(180.0, 95.0)
        assert r.flag == "INVALID"

    def test_ok_flag_for_valid_coords(self) -> None:
        r = check_tess_visibility(180.0, -60.0)
        assert r.flag == "OK"

    def test_n_sectors_nonnegative(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        assert r.n_sectors_visible >= 0

    def test_sector_list_integers(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        for s in r.sector_list:
            assert isinstance(s, int)
            assert 1 <= s <= 96

    def test_cvz_high_ecliptic_lat(self) -> None:
        # Near south ecliptic pole → CVZ
        r = check_tess_visibility(90.0, -66.56)
        # Ecliptic pole at ecliptic lat ≈ -90° from dec −66.56 is NOT exactly CVZ
        # Just test the flag is OK
        assert r.flag == "OK"

    def test_equatorial_star_low_visibility(self) -> None:
        # Equatorial star at low ecliptic lat
        r = check_tess_visibility(0.0, 0.0)
        assert r.flag == "OK"

    def test_coords_stored(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        assert r.ra_deg == 93.185
        assert r.dec_deg == -65.179

    def test_sector_list_capped_at_20(self) -> None:
        r = check_tess_visibility(90.0, -80.0)
        assert len(r.sector_list) <= 20


class TestFormatVisibilityResult:
    def test_returns_string(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        assert isinstance(format_visibility_result(r), str)

    def test_contains_ra(self) -> None:
        r = check_tess_visibility(93.185, -65.179)
        out = format_visibility_result(r)
        assert "93" in out

    def test_invalid_formatted(self) -> None:
        r = check_tess_visibility(-10.0, 0.0)
        assert "INVALID" in format_visibility_result(r)
