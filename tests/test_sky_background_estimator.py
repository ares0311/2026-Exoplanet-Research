"""Tests for Skills/sky_background_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sky_background_estimator import estimate_sky_background, format_sky_background


class TestEstimateSkyBackground:
    def test_basic_calculation(self) -> None:
        r = estimate_sky_background(10.0, 9, 1000.0)
        assert r.flag == "OK"
        assert r.total_sky_counts == 90.0

    def test_sky_fraction_correct(self) -> None:
        r = estimate_sky_background(100.0, 1, 900.0)
        assert abs(r.sky_fraction - 0.1) < 1e-9

    def test_no_sky_zero_fraction(self) -> None:
        r = estimate_sky_background(0.0, 9, 1000.0)
        assert r.sky_fraction == 0.0
        assert r.flag == "OK"

    def test_invalid_sky_counts(self) -> None:
        r = estimate_sky_background(-1.0, 9, 1000.0)
        assert r.flag == "INVALID_SKY_COUNTS"

    def test_invalid_n_pixels(self) -> None:
        r = estimate_sky_background(10.0, 0, 1000.0)
        assert r.flag == "INVALID_N_PIXELS"

    def test_invalid_target_counts(self) -> None:
        r = estimate_sky_background(10.0, 9, 0.0)
        assert r.flag == "INVALID_TARGET_COUNTS"

    def test_total_sky_counts_formula(self) -> None:
        r = estimate_sky_background(5.0, 4, 500.0)
        assert abs(r.total_sky_counts - 20.0) < 1e-9

    def test_sky_fraction_in_range(self) -> None:
        r = estimate_sky_background(50.0, 9, 1000.0)
        assert 0.0 <= r.sky_fraction <= 1.0

    def test_target_counts_stored(self) -> None:
        r = estimate_sky_background(10.0, 9, 1234.0)
        assert r.target_counts == 1234.0

    def test_n_pixels_stored(self) -> None:
        r = estimate_sky_background(10.0, 7, 1000.0)
        assert r.n_pixels == 7

    def test_result_frozen(self) -> None:
        r = estimate_sky_background(10.0, 9, 1000.0)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        r = estimate_sky_background(10.0, 9, 1000.0)
        s = format_sky_background(r)
        assert isinstance(s, str)
        assert "Sky" in s
