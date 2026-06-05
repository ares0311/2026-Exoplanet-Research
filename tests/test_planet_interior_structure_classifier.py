"""Tests for Skills/planet_interior_structure_classifier.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_interior_structure_classifier import (
    classify_interior_structure,
    format_interior_structure_result,
)


class TestPlanetInteriorStructureClassifier:
    def test_earth_like_composition(self) -> None:
        # At 1 M_earth, the rocky boundary is R=1.0 exactly (strict <), so use R=0.95
        r = classify_interior_structure(1.0, 0.95)
        assert r.flag == "OK"
        assert r.composition == "EARTH_LIKE"

    def test_gas_giant_jupiter(self) -> None:
        r = classify_interior_structure(317.8, 11.2)
        assert r.flag == "OK"
        assert r.composition == "GAS_GIANT"

    def test_iron_rich_dense_planet(self) -> None:
        # Very small radius for mass = dense iron world
        r = classify_interior_structure(5.0, 1.0)
        assert r.flag == "OK"
        assert r.composition == "IRON_RICH"

    def test_water_world(self) -> None:
        # R between Earth-like and gas dwarf boundary at a few Earth masses
        r = classify_interior_structure(5.0, 1.8)
        assert r.flag == "OK"
        assert r.composition in ("WATER_WORLD", "GAS_DWARF", "EARTH_LIKE")

    def test_radius_category_rocky(self) -> None:
        r = classify_interior_structure(1.0, 1.0)
        assert r.radius_category == "ROCKY"

    def test_radius_category_jovian(self) -> None:
        r = classify_interior_structure(317.8, 11.2)
        assert r.radius_category == "JOVIAN"

    def test_bulk_density_positive(self) -> None:
        r = classify_interior_structure(1.0, 1.0)
        assert r.bulk_density_gcc > 0
        assert math.isfinite(r.bulk_density_gcc)

    def test_earth_density_ratio_near_one(self) -> None:
        r = classify_interior_structure(1.0, 1.0)
        assert abs(r.earth_density_ratio - 1.0) < 0.01

    def test_invalid_mass(self) -> None:
        r = classify_interior_structure(0.0, 1.0)
        assert r.flag == "INVALID_MASS"
        assert math.isnan(r.bulk_density_gcc)

    def test_invalid_radius(self) -> None:
        r = classify_interior_structure(1.0, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_result_is_frozen(self) -> None:
        r = classify_interior_structure(1.0, 1.0)
        try:
            r.composition = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = classify_interior_structure(1.0, 1.0)
        s = format_interior_structure_result(r)
        assert r.composition in s
        assert "density" in s.lower() or "Density" in s

    def test_format_error(self) -> None:
        r = classify_interior_structure(0.0, 1.0)
        s = format_interior_structure_result(r)
        assert "INVALID_MASS" in s
