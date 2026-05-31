"""Tests for Skills/night_sky_brightness_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from night_sky_brightness_estimator import estimate_sky_brightness, format_sky_brightness_result


class TestSkyBrightnessEstimator:
    def test_new_moon_dark_sky(self) -> None:
        r = estimate_sky_brightness(0.0, 90.0)
        assert r.flag == "OK"
        assert r.sky_brightness_vmag == r.dark_sky_vmag

    def test_full_moon_brighter(self) -> None:
        r_dark = estimate_sky_brightness(0.0, 90.0)
        r_full = estimate_sky_brightness(1.0, 90.0)
        assert r_full.sky_brightness_vmag < r_dark.sky_brightness_vmag

    def test_close_moon_brighter_than_far(self) -> None:
        r_close = estimate_sky_brightness(0.5, 20.0)
        r_far = estimate_sky_brightness(0.5, 120.0)
        assert r_close.sky_brightness_vmag < r_far.sky_brightness_vmag

    def test_invalid_illumination_negative(self) -> None:
        r = estimate_sky_brightness(-0.1, 90.0)
        assert r.flag == "INVALID_ILLUMINATION"

    def test_invalid_illumination_over_one(self) -> None:
        r = estimate_sky_brightness(1.1, 90.0)
        assert r.flag == "INVALID_ILLUMINATION"

    def test_invalid_separation_negative(self) -> None:
        r = estimate_sky_brightness(0.5, -10.0)
        assert r.flag == "INVALID_SEPARATION"

    def test_sky_brightness_is_float(self) -> None:
        r = estimate_sky_brightness(0.5, 60.0)
        assert isinstance(r.sky_brightness_vmag, float)

    def test_excess_mag_non_negative_with_moon(self) -> None:
        r = estimate_sky_brightness(0.8, 30.0)
        assert r.excess_mag >= 0.0

    def test_excess_mag_zero_new_moon(self) -> None:
        r = estimate_sky_brightness(0.0, 90.0)
        assert r.excess_mag == 0.0

    def test_custom_dark_sky(self) -> None:
        r = estimate_sky_brightness(0.0, 90.0, dark_sky_vmag=21.5)
        assert r.dark_sky_vmag == 21.5
        assert r.sky_brightness_vmag == 21.5

    def test_half_moon_moderate_brightening(self) -> None:
        r = estimate_sky_brightness(0.5, 90.0)
        assert r.sky_brightness_vmag < 22.0

    def test_format_returns_string(self) -> None:
        r = estimate_sky_brightness(0.5, 60.0)
        s = format_sky_brightness_result(r)
        assert isinstance(s, str)
        assert "brightness" in s.lower()
