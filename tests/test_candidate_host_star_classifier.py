"""Tests for Skills/candidate_host_star_classifier.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_host_star_classifier import (
    classify_host_star,
    format_host_star_classification_result,
)


class TestClassifyHostStar:
    def test_ok_flag(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        assert r.flag == "OK"

    def test_sun_is_g_type(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        assert r.spectral_type.startswith("G")

    def test_sun_is_main_sequence(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        assert r.luminosity_class == "MAIN_SEQUENCE"

    def test_hot_a_star(self) -> None:
        r = classify_host_star(8000.0, 4.0)
        assert r.spectral_type.startswith("A")

    def test_cool_m_dwarf(self) -> None:
        r = classify_host_star(3500.0, 4.8)
        assert r.spectral_type.startswith("M")

    def test_giant_classification(self) -> None:
        r = classify_host_star(5000.0, 2.5)
        assert r.luminosity_class in ("GIANT", "SUBGIANT")

    def test_giant_risk_true_for_low_logg(self) -> None:
        r = classify_host_star(5000.0, 2.0)
        assert r.is_giant_risk

    def test_giant_risk_false_for_main_sequence(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        assert not r.is_giant_risk

    def test_pulsator_risk_for_delta_scuti(self) -> None:
        r = classify_host_star(7500.0, 4.0)
        assert r.is_pulsator_risk

    def test_no_pulsator_risk_for_solar(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        assert not r.is_pulsator_risk

    def test_tidal_risk_rapid_rotator(self) -> None:
        r = classify_host_star(5778.0, 4.44, vsini_kms=30.0)
        assert r.tidal_circularization_risk

    def test_invalid_teff(self) -> None:
        r = classify_host_star(0.0, 4.44)
        assert r.flag == "INVALID_TEFF"

    def test_format_returns_string(self) -> None:
        r = classify_host_star(5778.0, 4.44)
        s = format_host_star_classification_result(r)
        assert isinstance(s, str)
        assert r.flag in s
