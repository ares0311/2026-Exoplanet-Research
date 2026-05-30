"""Tests for Skills/transit_color_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_color_checker import check_transit_color, format_color_check_result


class TestCheckTransitColor:
    def test_uniform_not_chromatic(self) -> None:
        r = check_transit_color([1000.0, 1000.0, 1000.0])
        assert r.chromatic is False
        assert r.flag == "OK"

    def test_varying_chromatic(self) -> None:
        r = check_transit_color([1000.0, 500.0])
        assert r.chromatic is True
        assert r.flag == "CHROMATIC"

    def test_single_insufficient(self) -> None:
        r = check_transit_color([1000.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_mean_depth_correct(self) -> None:
        r = check_transit_color([1000.0, 2000.0])
        assert abs(r.mean_depth_ppm - 1500.0) < 1e-9

    def test_max_deviation_correct(self) -> None:
        r = check_transit_color([1000.0, 2000.0])
        assert abs(r.max_deviation_ppm - 500.0) < 1e-9

    def test_band_names_assigned(self) -> None:
        r = check_transit_color([1000.0, 1050.0], ["g", "r"])
        assert "g" in r.band_names or "r" in r.band_names

    def test_band_name_mismatch(self) -> None:
        r = check_transit_color([1000.0, 1050.0], ["g"])
        assert r.flag == "BAND_NAME_MISMATCH"

    def test_invalid_depth(self) -> None:
        r = check_transit_color([-100.0, 1000.0])
        assert r.flag == "INVALID_DEPTH"

    def test_custom_threshold(self) -> None:
        r = check_transit_color([1000.0, 1200.0], chromaticity_threshold=0.05)
        assert r.chromatic is True

    def test_relative_variation_stored(self) -> None:
        r = check_transit_color([900.0, 1100.0])
        assert r.relative_variation >= 0

    def test_format_returns_string(self) -> None:
        r = check_transit_color([1000.0, 1050.0])
        s = format_color_check_result(r)
        assert isinstance(s, str)
        assert "Color" in s

    def test_empty_insufficient(self) -> None:
        r = check_transit_color([])
        assert r.flag == "INSUFFICIENT_DATA"
