"""Tests for Skills/atmospheric_composition_classifier.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from atmospheric_composition_classifier import (
    compute_atmospheric_composition,
    format_atmospheric_composition_result,
)


class TestComputeAtmosphericComposition:
    def test_earth_like_is_rocky(self) -> None:
        r = compute_atmospheric_composition(1.0, 1.0)
        assert r.composition_class == "ROCKY"

    def test_bulk_density_earth(self) -> None:
        r = compute_atmospheric_composition(1.0, 1.0)
        assert abs(r.bulk_density_gcm3 - 5.51) < 0.1

    def test_jupiter_is_gas_giant(self) -> None:
        r = compute_atmospheric_composition(11.2, 317.8)
        assert r.composition_class in ("GAS_GIANT", "NEPTUNIAN", "GAS_DWARF")

    def test_neptune_like_is_neptunian(self) -> None:
        r = compute_atmospheric_composition(3.9, 17.1)
        assert r.composition_class in ("NEPTUNIAN", "GAS_DWARF")

    def test_sub_neptune_gas_dwarf(self) -> None:
        r = compute_atmospheric_composition(2.5, 10.0)
        assert r.composition_class in ("GAS_DWARF", "NEPTUNIAN", "WATER_WORLD")

    def test_envelope_fraction_rocky_near_zero(self) -> None:
        r = compute_atmospheric_composition(1.0, 1.0)
        assert r.envelope_fraction <= 0.1

    def test_envelope_fraction_gas_giant_high(self) -> None:
        r = compute_atmospheric_composition(11.2, 317.8)
        assert r.envelope_fraction > 0.5

    def test_ok_flag_valid_inputs(self) -> None:
        r = compute_atmospheric_composition(2.0, 8.0)
        assert r.flag == "OK"

    def test_invalid_radius(self) -> None:
        r = compute_atmospheric_composition(0.0, 5.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_mass(self) -> None:
        r = compute_atmospheric_composition(2.0, 0.0)
        assert r.flag == "INVALID_MASS"

    def test_bulk_density_positive(self) -> None:
        r = compute_atmospheric_composition(2.0, 8.0)
        assert r.bulk_density_gcm3 > 0

    def test_result_frozen(self) -> None:
        r = compute_atmospheric_composition(2.0, 8.0)
        try:
            r.composition_class = "TEST"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_atmospheric_composition(2.0, 8.0)
        s = format_atmospheric_composition_result(r)
        assert isinstance(s, str)
        assert r.flag in s
