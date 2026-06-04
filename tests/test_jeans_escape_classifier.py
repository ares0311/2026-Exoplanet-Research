"""Tests for Skills/jeans_escape_classifier.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from jeans_escape_classifier import compute_jeans_escape, format_jeans_escape_result


class TestComputeJeansEscape:
    def test_ok_flag(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0)
        assert r.flag == "OK"

    def test_earth_nitrogen_stable(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 300.0, mean_molecular_weight_amu=28.0)
        assert r.retention_class == "STABLE"

    def test_hydrogen_hot_rapid_escape(self) -> None:
        r = compute_jeans_escape(0.1, 2.0, 5000.0, mean_molecular_weight_amu=2.0)
        assert r.retention_class in ("RAPID_ESCAPE", "MARGINAL")

    def test_jeans_parameter_positive(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0)
        assert r.jeans_parameter > 0.0

    def test_surface_gravity_positive(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0)
        assert r.surface_gravity_ms2 > 0.0

    def test_thermal_velocity_positive(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0)
        assert r.thermal_velocity_ms > 0.0

    def test_larger_mass_more_stable(self) -> None:
        r_small = compute_jeans_escape(0.1, 1.0, 1000.0)
        r_large = compute_jeans_escape(10.0, 1.0, 1000.0)
        assert r_large.jeans_parameter > r_small.jeans_parameter

    def test_higher_temp_less_stable(self) -> None:
        r_cool = compute_jeans_escape(1.0, 1.0, 500.0)
        r_hot = compute_jeans_escape(1.0, 1.0, 5000.0)
        assert r_hot.jeans_parameter < r_cool.jeans_parameter

    def test_invalid_mass(self) -> None:
        r = compute_jeans_escape(0.0, 1.0, 1000.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_radius(self) -> None:
        r = compute_jeans_escape(1.0, 0.0, 1000.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_temp(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 0.0)
        assert r.flag == "INVALID_TEMP"

    def test_invalid_mu(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0, mean_molecular_weight_amu=0.0)
        assert r.flag == "INVALID_MEAN_MOLECULAR_WEIGHT"

    def test_format_returns_string(self) -> None:
        r = compute_jeans_escape(1.0, 1.0, 1000.0)
        s = format_jeans_escape_result(r)
        assert isinstance(s, str)
        assert r.flag in s
