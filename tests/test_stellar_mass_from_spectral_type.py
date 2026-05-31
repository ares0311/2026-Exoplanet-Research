"""Tests for Skills/stellar_mass_from_spectral_type.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_mass_from_spectral_type import estimate_mass_from_spectral_type, format_mass_result


class TestStellarMassFromSpectralType:
    def test_sun_g2v(self) -> None:
        r = estimate_mass_from_spectral_type("G2V")
        assert r.flag == "OK"
        assert r.spectral_class == "G"
        assert r.mass_min_msun < 1.0 < r.mass_max_msun

    def test_m_dwarf(self) -> None:
        r = estimate_mass_from_spectral_type("M4V")
        assert r.flag == "OK"
        assert r.mass_typical_msun < 0.5

    def test_o_star(self) -> None:
        r = estimate_mass_from_spectral_type("O5")
        assert r.flag == "OK"
        assert r.mass_min_msun > 10.0

    def test_b_star(self) -> None:
        r = estimate_mass_from_spectral_type("B3V")
        assert r.flag == "OK"
        assert r.mass_typical_msun > 2.0

    def test_a_star(self) -> None:
        r = estimate_mass_from_spectral_type("A0V")
        assert r.flag == "OK"
        assert r.mass_typical_msun > 1.4

    def test_k_star(self) -> None:
        r = estimate_mass_from_spectral_type("K5V")
        assert r.flag == "OK"
        assert r.mass_typical_msun < 1.0

    def test_luminosity_class_parsed(self) -> None:
        r = estimate_mass_from_spectral_type("G2V")
        assert r.luminosity_class == "V"

    def test_unknown_spectral_class(self) -> None:
        r = estimate_mass_from_spectral_type("X5V")
        assert r.flag == "UNKNOWN_SPECTRAL_CLASS"
        assert math.isnan(r.mass_typical_msun)

    def test_empty_spectral_type(self) -> None:
        r = estimate_mass_from_spectral_type("")
        assert r.flag == "INVALID_SPECTRAL_TYPE"

    def test_mass_min_lt_max(self) -> None:
        r = estimate_mass_from_spectral_type("F5V")
        assert r.mass_min_msun < r.mass_max_msun

    def test_mass_typical_in_range(self) -> None:
        r = estimate_mass_from_spectral_type("F5V")
        assert r.mass_min_msun <= r.mass_typical_msun <= r.mass_max_msun

    def test_format_returns_string(self) -> None:
        r = estimate_mass_from_spectral_type("G2V")
        s = format_mass_result(r)
        assert isinstance(s, str)
        assert "Mass" in s
