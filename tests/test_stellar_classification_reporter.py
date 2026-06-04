"""Tests for Skills/stellar_classification_reporter.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_classification_reporter import (
    build_stellar_classification_report,
    format_classification_report,
)


class TestBuildStellarClassificationReport:
    def test_sun_like_star(self) -> None:
        r = build_stellar_classification_report(teff_k=5778.0, logg=4.44)
        assert r.spectral_class == "G"
        assert r.flag == "OK"

    def test_hot_star(self) -> None:
        r = build_stellar_classification_report(teff_k=25000.0, logg=4.0)
        assert r.spectral_class == "B"

    def test_cool_m_dwarf(self) -> None:
        r = build_stellar_classification_report(teff_k=3500.0, logg=4.7)
        assert r.spectral_class == "M"
        assert r.flag == "OK"

    def test_k_dwarf(self) -> None:
        r = build_stellar_classification_report(teff_k=4800.0, logg=4.5)
        assert r.spectral_class == "K"

    def test_f_star(self) -> None:
        r = build_stellar_classification_report(teff_k=6500.0, logg=4.2)
        assert r.spectral_class in ("F", "G")  # boundary at 6000K

    def test_giant_logg(self) -> None:
        r = build_stellar_classification_report(teff_k=5000.0, logg=2.5)
        assert r.luminosity_class == "III"
        assert r.evolution_stage == "giant"

    def test_supergiant_logg(self) -> None:
        r = build_stellar_classification_report(teff_k=5000.0, logg=0.3)
        assert r.luminosity_class == "I"
        assert r.evolution_stage == "supergiant"

    def test_luminosity_lsun_positive(self) -> None:
        r = build_stellar_classification_report(teff_k=5778.0, logg=4.44)
        assert r.luminosity_lsun is not None
        assert r.luminosity_lsun > 0

    def test_radius_rsun_passed_through(self) -> None:
        r = build_stellar_classification_report(teff_k=5778.0, logg=4.44, radius_rsun=1.0)
        assert r.radius_rsun == 1.0

    def test_invalid_teff_zero(self) -> None:
        r = build_stellar_classification_report(teff_k=0.0, logg=4.4)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_logg_nan(self) -> None:
        r = build_stellar_classification_report(teff_k=5778.0, logg=float("nan"))
        assert r.flag == "INVALID_LOGG"

    def test_format_output(self) -> None:
        r = build_stellar_classification_report(teff_k=5778.0, logg=4.44)
        s = format_classification_report(r)
        assert "Spectral class" in s
        assert "|" in s
