"""Tests for Skills/stellar_companion_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_companion_checker import (
    check_stellar_companion,
    format_companion_result,
)


class TestCheckStellarCompanion:
    def test_within_limit(self) -> None:
        r = check_stellar_companion(10.0, 12.0, 2.0)
        assert r.companion_within_limit is True
        assert r.flag == "WITHIN_LIMIT"

    def test_outside_limit(self) -> None:
        r = check_stellar_companion(200.0, 12.0, 0.0)
        assert r.companion_within_limit is False
        assert r.flag == "OUTSIDE_LIMIT"

    def test_zero_separation(self) -> None:
        r = check_stellar_companion(0.0, 12.0, 0.0)
        assert r.companion_within_limit is True

    def test_negative_separation_invalid(self) -> None:
        r = check_stellar_companion(-1.0, 12.0, 0.0)
        assert r.flag == "INVALID_SEPARATION"
        assert r.companion_within_limit is False

    def test_negative_contrast_invalid(self) -> None:
        r = check_stellar_companion(5.0, 12.0, -1.0)
        assert r.flag == "INVALID_CONTRAST"

    def test_negative_tmag_invalid(self) -> None:
        r = check_stellar_companion(5.0, -1.0, 0.0)
        assert r.flag == "INVALID_TMAG"

    def test_detection_limit_formula(self) -> None:
        r = check_stellar_companion(0.0, 10.0, 5.0, pixel_scale_arcsec=21.0)
        # limit = 21.0 * (1.0 + 5.0/5.0) = 42.0
        assert abs(r.detection_limit_arcsec - 42.0) < 1e-9

    def test_zero_contrast_gives_pixel_scale(self) -> None:
        r = check_stellar_companion(0.0, 10.0, 0.0, pixel_scale_arcsec=21.0)
        assert abs(r.detection_limit_arcsec - 21.0) < 1e-9

    def test_custom_pixel_scale(self) -> None:
        r = check_stellar_companion(0.0, 10.0, 0.0, pixel_scale_arcsec=10.0)
        assert abs(r.detection_limit_arcsec - 10.0) < 1e-9

    def test_separation_stored(self) -> None:
        r = check_stellar_companion(15.5, 12.0, 1.0)
        assert r.separation_arcsec == 15.5

    def test_result_is_frozen(self) -> None:
        r = check_stellar_companion(5.0, 12.0, 1.0)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        r = check_stellar_companion(5.0, 12.0, 1.0)
        s = format_companion_result(r)
        assert isinstance(s, str)
        assert "WITHIN_LIMIT" in s or "OUTSIDE_LIMIT" in s
