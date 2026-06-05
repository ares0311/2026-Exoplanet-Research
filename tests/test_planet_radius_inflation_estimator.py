"""Tests for Skills/planet_radius_inflation_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_radius_inflation_estimator import (
    compute_radius_inflation,
    format_radius_inflation_result,
)


class TestComputeRadiusInflation:
    def test_ok_flag(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        assert r.flag == "OK"

    def test_stellar_flux_positive(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        assert r.stellar_flux_wm2 > 0.0

    def test_expected_radius_positive(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        assert r.expected_radius_rjup > 0.0

    def test_inflated_radius_ge_expected(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        assert r.inflated_radius_rjup >= r.expected_radius_rjup

    def test_closer_orbit_more_inflation(self) -> None:
        r_far = compute_radius_inflation(1.0, 1.2, 0.5)
        r_close = compute_radius_inflation(1.0, 1.2, 0.03)
        assert r_close.inflated_radius_rjup >= r_far.inflated_radius_rjup

    def test_inflation_class_set(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        assert r.inflation_class in ("NONE", "MILD", "MODERATE", "SEVERE")

    def test_distant_orbit_no_inflation(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 5.0)
        assert r.inflation_class == "NONE"

    def test_very_close_orbit_inflated(self) -> None:
        r = compute_radius_inflation(1.0, 1.8, 0.02, stellar_teff_k=6500.0)
        assert r.inflation_class in ("MODERATE", "SEVERE", "MILD")

    def test_invalid_mass(self) -> None:
        r = compute_radius_inflation(0.0, 1.2, 0.05)
        assert r.flag == "INVALID_MASS"

    def test_invalid_radius(self) -> None:
        r = compute_radius_inflation(1.0, 0.0, 0.05)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_distance(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_result_frozen(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        try:
            r.inflation_rjup = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_radius_inflation(1.0, 1.2, 0.05)
        s = format_radius_inflation_result(r)
        assert isinstance(s, str)
        assert r.flag in s
