"""Tests for Skills/stellar_evolution_classifier.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_evolution_classifier import classify_evolution_stage, format_evolution_result


class TestStellarEvolutionClassifier:
    def test_sun_is_main_sequence(self) -> None:
        r = classify_evolution_stage(5778.0, 4.44)
        assert r.stage == "main_sequence"
        assert r.spectral_class == "G"

    def test_giant_low_logg(self) -> None:
        r = classify_evolution_stage(4800.0, 2.5)
        assert r.stage == "giant"

    def test_subgiant(self) -> None:
        r = classify_evolution_stage(5500.0, 3.8)
        assert r.stage == "subgiant"

    def test_white_dwarf_high_logg(self) -> None:
        r = classify_evolution_stage(15000.0, 8.0)
        assert r.stage == "white_dwarf"

    def test_supergiant_very_low_logg(self) -> None:
        r = classify_evolution_stage(4000.0, 0.5)
        assert r.stage == "supergiant"

    def test_spectral_class_o(self) -> None:
        r = classify_evolution_stage(40000.0, 4.5)
        assert r.spectral_class == "O"

    def test_spectral_class_m(self) -> None:
        r = classify_evolution_stage(3200.0, 4.8)
        assert r.spectral_class == "M"

    def test_invalid_teff(self) -> None:
        r = classify_evolution_stage(0.0, 4.44)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_logg(self) -> None:
        import math
        r = classify_evolution_stage(5778.0, math.nan)
        assert r.flag == "INVALID_LOGG"

    def test_luminosity_inconsistent_giant(self) -> None:
        # Giant with very low luminosity — inconsistent
        r = classify_evolution_stage(4800.0, 2.5, luminosity_lsun=0.1)
        assert r.flag == "LUMINOSITY_INCONSISTENT"

    def test_ok_with_consistent_luminosity(self) -> None:
        r = classify_evolution_stage(5778.0, 4.44, luminosity_lsun=1.0)
        assert r.flag == "OK"

    def test_format_returns_string(self) -> None:
        r = classify_evolution_stage(5778.0, 4.44)
        s = format_evolution_result(r)
        assert isinstance(s, str)
        assert "stage" in s.lower()
