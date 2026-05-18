"""Tests for Skills.planet_radius_estimator."""
from __future__ import annotations

import pytest
from Skills.planet_radius_estimator import (
    PlanetRadiusResult,
    estimate_planet_radius,
    format_planet_radius_result,
)


class TestEstimatePlanetRadius:
    def test_returns_result(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0)
        assert isinstance(r, PlanetRadiusResult)

    def test_radius_ratio_correct(self) -> None:
        import math
        r = estimate_planet_radius(10000.0, 1.0)
        assert r.radius_ratio == pytest.approx(math.sqrt(10000.0 / 1e6), rel=1e-4)

    def test_rearth_uses_stellar_radius(self) -> None:
        r1 = estimate_planet_radius(1000.0, 1.0)
        r2 = estimate_planet_radius(1000.0, 2.0)
        assert r2.planet_radius_rearth == pytest.approx(r1.planet_radius_rearth * 2.0, rel=1e-4)

    def test_zero_depth_gives_zero_radius(self) -> None:
        r = estimate_planet_radius(0.0, 1.0)
        assert r.planet_radius_rearth == pytest.approx(0.0, abs=1e-6)

    def test_classification_sub_earth(self) -> None:
        r = estimate_planet_radius(50.0, 1.0)
        assert r.classification == "sub-Earth"

    def test_classification_super_earth(self) -> None:
        # ~1.5 R_earth: depth = (1.5 / 109.076)^2 * 1e6 ≈ 189 ppm
        r = estimate_planet_radius(189.0, 1.0)
        assert r.classification == "super-Earth"

    def test_classification_neptune(self) -> None:
        # ~4 R_earth: depth = (4/109.076)^2 * 1e6 ≈ 1342 ppm
        r = estimate_planet_radius(1342.0, 1.0)
        assert r.classification == "Neptune"

    def test_classification_jupiter(self) -> None:
        # ~11 R_earth: depth = (11/109.076)^2 * 1e6 ≈ 10159 ppm
        r = estimate_planet_radius(10159.0, 1.0)
        assert r.classification == "Jupiter"

    def test_classification_stellar(self) -> None:
        # Very large radius
        r = estimate_planet_radius(100000.0, 1.0)
        assert r.classification == "stellar"

    def test_error_propagation_depth_only(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0, depth_err_ppm=100.0)
        assert r.planet_radius_err_rearth is not None
        assert r.planet_radius_err_rearth > 0

    def test_error_propagation_stellar_only(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0, stellar_radius_err_rsun=0.1)
        assert r.planet_radius_err_rearth is not None
        assert r.planet_radius_err_rearth > 0

    def test_no_error_when_no_inputs(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0)
        assert r.planet_radius_err_rearth is None

    def test_rjup_conversion(self) -> None:
        r = estimate_planet_radius(10000.0, 1.0)
        assert r.planet_radius_rjup == pytest.approx(r.planet_radius_rsun * 9.731, rel=1e-4)


class TestFormatPlanetRadius:
    def test_returns_string(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0)
        assert isinstance(format_planet_radius_result(r), str)

    def test_contains_classification(self) -> None:
        r = estimate_planet_radius(1000.0, 1.0)
        assert r.classification in format_planet_radius_result(r)
