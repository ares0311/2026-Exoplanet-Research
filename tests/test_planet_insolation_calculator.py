"""Tests for Skills/planet_insolation_calculator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_insolation_calculator import compute_insolation, format_insolation_result


class TestComputeInsolation:
    def test_earth_insolation(self) -> None:
        # L=1, a=1 AU => S=1 S_earth
        r = compute_insolation(1.0, 1.0)
        assert r.flag == "OK"
        assert abs(r.insolation_earth - 1.0) < 1e-6

    def test_earth_hz_class(self) -> None:
        r = compute_insolation(1.0, 1.0)
        assert r.hz_class == "HABITABLE_ZONE"

    def test_hot_jupiter_too_hot(self) -> None:
        r = compute_insolation(1.0, 0.05)
        assert r.hz_class == "TOO_HOT"

    def test_cold_outer_planet(self) -> None:
        r = compute_insolation(1.0, 10.0)
        assert r.hz_class == "TOO_COLD"

    def test_invalid_luminosity(self) -> None:
        r = compute_insolation(0.0, 1.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_invalid_sma(self) -> None:
        r = compute_insolation(1.0, 0.0)
        assert r.flag == "INVALID_SMA"

    def test_insolation_inverse_square(self) -> None:
        r1 = compute_insolation(1.0, 1.0)
        r2 = compute_insolation(1.0, 2.0)
        assert abs(r2.insolation_earth - r1.insolation_earth / 4.0) < 1e-5

    def test_higher_luminosity_more_flux(self) -> None:
        r1 = compute_insolation(1.0, 1.0)
        r2 = compute_insolation(2.0, 1.0)
        assert r2.insolation_earth > r1.insolation_earth

    def test_insolation_is_float(self) -> None:
        r = compute_insolation(1.0, 1.0)
        assert isinstance(r.insolation_earth, float)

    def test_teff_affects_hz_boundaries(self) -> None:
        # Different Teff shifts HZ; just check it runs
        r = compute_insolation(0.1, 0.3, teff_k=3700.0)
        assert r.flag == "OK"

    def test_format_returns_string(self) -> None:
        r = compute_insolation(1.0, 1.0)
        s = format_insolation_result(r)
        assert isinstance(s, str)
        assert "Insolation" in s

    def test_negative_sma_invalid(self) -> None:
        r = compute_insolation(1.0, -1.0)
        assert r.flag == "INVALID_SMA"
