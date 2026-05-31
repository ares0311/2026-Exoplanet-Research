"""Tests for Skills/roche_lobe_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from roche_lobe_checker import check_roche_lobe, format_roche_lob_result


class TestRocheLobe:
    def test_earth_not_eb(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        assert r.flag == "OK"
        assert r.eb_suspected is False

    def test_fill_factor_positive(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        assert r.fill_factor > 0.0

    def test_roche_radius_positive(self) -> None:
        r = check_roche_lobe(10.0, 100.0, 1.0, 0.05)
        assert r.roche_radius_rearth > 0.0

    def test_hill_radius_positive(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        assert r.hill_radius_rearth > 0.0

    def test_invalid_planet_radius(self) -> None:
        r = check_roche_lobe(0.0, 1.0, 1.0, 1.0)
        assert r.flag == "INVALID_PLANET_RADIUS"

    def test_invalid_planet_mass(self) -> None:
        r = check_roche_lobe(1.0, 0.0, 1.0, 1.0)
        assert r.flag == "INVALID_PLANET_MASS"

    def test_invalid_stellar_mass(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 0.0, 1.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_invalid_sma(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 0.0)
        assert r.flag == "INVALID_SMA"

    def test_eb_suspected_large_radius(self) -> None:
        # Very large planet near the star
        r = check_roche_lobe(100000.0, 1e6, 1.0, 0.001)
        assert r.eb_suspected is True
        assert r.flag == "EB_SUSPECTED"

    def test_hill_radius_larger_than_roche(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        assert r.hill_radius_rearth > r.roche_radius_rearth

    def test_fill_factor_lt_1_for_planet(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        assert r.fill_factor < 1.0

    def test_format_returns_string(self) -> None:
        r = check_roche_lobe(1.0, 1.0, 1.0, 1.0)
        s = format_roche_lob_result(r)
        assert isinstance(s, str)
        assert "Fill factor" in s
