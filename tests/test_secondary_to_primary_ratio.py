"""Tests for Skills/secondary_to_primary_ratio.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from secondary_to_primary_ratio import compute_secondary_to_primary_ratio, format_ratio_result


class TestComputeSecondaryToPrimaryRatio:
    def test_planet_consistent(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 50.0)
        assert r.planet_consistent is True
        assert r.flag == "PLANET_CONSISTENT"

    def test_eb_suspected(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 8000.0)
        assert r.planet_consistent is False
        assert r.flag == "EB_SUSPECTED"

    def test_zero_secondary(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 0.0)
        assert r.planet_consistent is True
        assert r.ratio == 0.0

    def test_invalid_primary(self) -> None:
        r = compute_secondary_to_primary_ratio(0.0, 100.0)
        assert r.flag == "INVALID_PRIMARY_DEPTH"

    def test_invalid_secondary_negative(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, -50.0)
        assert r.flag == "INVALID_SECONDARY_DEPTH"

    def test_ratio_correct(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 200.0)
        assert abs(r.ratio - 0.02) < 1e-9

    def test_custom_albedo_max(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 500.0, albedo_max=0.03)
        assert r.planet_consistent is False

    def test_primary_depth_stored(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 200.0)
        assert r.primary_depth_ppm == 10000.0

    def test_secondary_depth_stored(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 200.0)
        assert r.secondary_depth_ppm == 200.0

    def test_ratio_float(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 200.0)
        assert isinstance(r.ratio, float)

    def test_format_returns_string(self) -> None:
        r = compute_secondary_to_primary_ratio(10000.0, 200.0)
        s = format_ratio_result(r)
        assert isinstance(s, str)
        assert "Depth" in s

    def test_negative_primary_invalid(self) -> None:
        r = compute_secondary_to_primary_ratio(-100.0, 50.0)
        assert r.flag == "INVALID_PRIMARY_DEPTH"
