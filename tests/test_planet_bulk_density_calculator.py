"""Tests for Skills/planet_bulk_density_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_bulk_density_calculator import (
    BulkDensityResult,
    compute_bulk_density,
    format_bulk_density_result,
)


class TestBulkDensityResult:
    def test_dataclass_fields(self):
        r = BulkDensityResult(density_gcc=5.5, composition_hint="rocky", flag="OK")
        assert r.density_gcc == 5.5
        assert r.composition_hint == "rocky"

    def test_frozen(self):
        r = BulkDensityResult(density_gcc=5.5, composition_hint="rocky")
        try:
            r.density_gcc = 0
            raise AssertionError("Should be frozen")
        except Exception:
            pass


class TestComputeBulkDensity:
    def test_earth_like_density(self):
        # Earth: 1 M_earth, 1 R_earth → ~5.51 g/cm^3
        r = compute_bulk_density(1.0, 1.0)
        assert abs(r.density_gcc - 5.51) < 0.1
        assert r.flag == "OK"

    def test_rocky_hint(self):
        r = compute_bulk_density(1.0, 1.0)
        assert r.composition_hint == "rocky"

    def test_iron_hint(self):
        # Heavy small planet → iron
        r = compute_bulk_density(5.0, 1.0)
        assert r.composition_hint == "iron"

    def test_water_world_hint(self):
        # Light large planet → water world
        r = compute_bulk_density(5.0, 3.0)
        assert r.composition_hint in ("water_world", "gas_dwarf", "rocky")

    def test_gas_dwarf_hint(self):
        # Very low density
        r = compute_bulk_density(1.0, 10.0)
        assert r.composition_hint == "gas_dwarf"

    def test_zero_mass_error(self):
        r = compute_bulk_density(0.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_radius_error(self):
        r = compute_bulk_density(1.0, 0.0)
        assert r.flag == "ERROR"

    def test_negative_mass_error(self):
        r = compute_bulk_density(-1.0, 1.0)
        assert r.flag == "ERROR"

    def test_density_scales_with_radius_cubed(self):
        r1 = compute_bulk_density(1.0, 1.0)
        r2 = compute_bulk_density(1.0, 2.0)
        # density should decrease by factor 8 when radius doubles
        assert abs(r1.density_gcc / r2.density_gcc - 8.0) < 0.1

    def test_density_scales_linearly_with_mass(self):
        r1 = compute_bulk_density(1.0, 1.0)
        r2 = compute_bulk_density(2.0, 1.0)
        assert abs(r2.density_gcc / r1.density_gcc - 2.0) < 0.01


class TestFormatBulkDensity:
    def test_returns_string(self):
        r = compute_bulk_density(1.0, 1.0)
        s = format_bulk_density_result(r)
        assert isinstance(s, str)

    def test_contains_composition(self):
        r = compute_bulk_density(1.0, 1.0)
        s = format_bulk_density_result(r)
        assert r.composition_hint in s
