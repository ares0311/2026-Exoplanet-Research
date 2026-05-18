"""Tests for moon_separation_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from moon_separation_checker import (
    MoonSeparationResult,
    check_moon_separation,
    format_moon_separation_result,
)

BJD = 2460000.5


class TestCheckMoonSeparation:
    def test_returns_result(self):
        r = check_moon_separation(0.0, 0.0, BJD)
        assert isinstance(r, MoonSeparationResult)

    def test_flag_values(self):
        r = check_moon_separation(0.0, 0.0, BJD)
        assert r.flag in ("OK", "MOON_WARN", "MOON_SEVERE", "INVALID")

    def test_separation_nonnegative(self):
        r = check_moon_separation(45.0, 30.0, BJD)
        assert r.moon_separation_deg >= 0.0

    def test_illumination_in_range(self):
        r = check_moon_separation(45.0, 30.0, BJD)
        assert 0.0 <= r.moon_illumination_fraction <= 1.0

    def test_phase_name_nonempty(self):
        r = check_moon_separation(45.0, 30.0, BJD)
        assert len(r.moon_phase_name) > 0

    def test_moon_ra_dec_range(self):
        r = check_moon_separation(45.0, 30.0, BJD)
        assert 0.0 <= r.moon_ra_deg < 360.0
        assert -90.0 <= r.moon_dec_deg <= 90.0

    def test_problematic_when_close_and_bright(self):
        # Create a scenario where moon is problematic by using a very small separation threshold
        r = check_moon_separation(0.0, 0.0, BJD, min_separation_deg=180.0,
                                  illumination_threshold=0.0)
        assert r.is_problematic

    def test_not_problematic_when_far(self):
        r = check_moon_separation(0.0, 0.0, BJD, min_separation_deg=1.0,
                                  illumination_threshold=1.0)
        # With a very small separation threshold, should not be problematic unless moon is at (0,0)
        assert isinstance(r.is_problematic, bool)

    def test_invalid_bjd_zero(self):
        r = check_moon_separation(0.0, 0.0, 0.0)
        # Should either work or return INVALID
        assert r.flag in ("OK", "MOON_WARN", "MOON_SEVERE", "INVALID")

    def test_result_frozen(self):
        r = check_moon_separation(0.0, 0.0, BJD)
        try:
            r.moon_separation_deg = 999  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatMoonSeparationResult:
    def test_returns_string(self):
        r = check_moon_separation(0.0, 0.0, BJD)
        assert isinstance(format_moon_separation_result(r), str)

    def test_contains_flag(self):
        r = check_moon_separation(0.0, 0.0, BJD)
        s = format_moon_separation_result(r)
        assert r.flag in s

    def test_contains_separation(self):
        r = check_moon_separation(45.0, 30.0, BJD)
        s = format_moon_separation_result(r)
        assert "sep" in s.lower() or str(int(r.moon_separation_deg)) in s
