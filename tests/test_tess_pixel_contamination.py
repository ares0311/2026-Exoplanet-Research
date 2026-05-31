"""Tests for Skills/tess_pixel_contamination.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tess_pixel_contamination import estimate_pixel_contamination, format_pixel_contamination


class TestTessPixelContamination:
    def test_no_contamination(self) -> None:
        r = estimate_pixel_contamination(1.0)
        assert r.flag == "OK"
        assert abs(r.contamination_fraction) < 0.01

    def test_depth_factor_unity_for_crowdsap1(self) -> None:
        r = estimate_pixel_contamination(1.0)
        assert abs(r.corrected_depth_factor - 1.0) < 1e-6

    def test_moderate_contamination(self) -> None:
        r = estimate_pixel_contamination(0.7)
        assert r.flag == "OK"
        assert abs(r.contamination_fraction - 0.3) < 1e-4

    def test_high_contamination_flag(self) -> None:
        r = estimate_pixel_contamination(0.3)
        assert r.flag == "HIGH_CONTAMINATION"

    def test_depth_factor_gt1_for_contamination(self) -> None:
        r = estimate_pixel_contamination(0.8)
        assert r.corrected_depth_factor > 1.0

    def test_neighbour_dilution_increases_contamination(self) -> None:
        r1 = estimate_pixel_contamination(0.9, neighbour_flux_ratio=0.0)
        r2 = estimate_pixel_contamination(0.9, neighbour_flux_ratio=0.05)
        assert r2.contamination_fraction > r1.contamination_fraction

    def test_invalid_crowdsap_zero(self) -> None:
        r = estimate_pixel_contamination(0.0)
        assert r.flag == "INVALID_CROWDSAP"
        assert math.isnan(r.corrected_depth_factor)

    def test_invalid_crowdsap_negative(self) -> None:
        r = estimate_pixel_contamination(-0.1)
        assert r.flag == "INVALID_CROWDSAP"

    def test_invalid_neighbour_flux(self) -> None:
        r = estimate_pixel_contamination(0.8, neighbour_flux_ratio=-0.1)
        assert r.flag == "INVALID_NEIGHBOUR_FLUX"

    def test_invalid_depth(self) -> None:
        r = estimate_pixel_contamination(0.8, depth_ppm=-10.0)
        assert r.flag == "INVALID_DEPTH"

    def test_crowdsap_stored(self) -> None:
        r = estimate_pixel_contamination(0.75)
        assert r.crowdsap == 0.75

    def test_format_returns_string(self) -> None:
        r = estimate_pixel_contamination(0.8)
        s = format_pixel_contamination(r)
        assert isinstance(s, str)
        assert "CROWDSAP" in s
