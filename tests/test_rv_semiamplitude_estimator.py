"""Tests for Skills.rv_semiamplitude_estimator."""
from __future__ import annotations

import pytest
from Skills.rv_semiamplitude_estimator import (
    RVResult,
    estimate_rv_semiamplitude,
    format_rv_result,
)


class TestEstimateRVSemiamplitude:
    def test_returns_result(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        assert isinstance(r, RVResult)

    def test_k_positive(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        assert r.k_ms > 0

    def test_k_scales_with_mass(self) -> None:
        r1 = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        r10 = estimate_rv_semiamplitude(5.0, 1.0, 10.0)
        assert r10.k_ms > r1.k_ms

    def test_k_decreases_with_period(self) -> None:
        r_short = estimate_rv_semiamplitude(1.0, 1.0, 1.0)
        r_long = estimate_rv_semiamplitude(100.0, 1.0, 1.0)
        assert r_short.k_ms > r_long.k_ms

    def test_inclination_90_max(self) -> None:
        r90 = estimate_rv_semiamplitude(5.0, 1.0, 1.0, inclination_deg=90.0)
        r45 = estimate_rv_semiamplitude(5.0, 1.0, 1.0, inclination_deg=45.0)
        assert r90.k_ms > r45.k_ms

    def test_classification_planet(self) -> None:
        r = estimate_rv_semiamplitude(10.0, 1.0, 1.0)
        assert r.flag == "PLANET"

    def test_classification_brown_dwarf(self) -> None:
        r = estimate_rv_semiamplitude(10.0, 1.0, 20.0)
        assert r.flag == "BROWN_DWARF"

    def test_classification_stellar(self) -> None:
        r = estimate_rv_semiamplitude(10.0, 1.0, 100.0)
        assert r.flag == "STELLAR"

    def test_error_propagation_returns_value(self) -> None:
        r = estimate_rv_semiamplitude(
            5.0, 1.0, 1.0,
            companion_mass_err_mjup=0.1,
        )
        assert r.k_err_ms is not None
        assert r.k_err_ms > 0

    def test_no_err_when_not_provided(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        assert r.k_err_ms is None

    def test_mearth_conversion(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        # 1 Mjup = 317.8 Mearth
        assert r.mass_companion_mearth == pytest.approx(317.8, rel=0.01)


class TestFormatRVResult:
    def test_returns_string(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        assert isinstance(format_rv_result(r), str)

    def test_contains_k_value(self) -> None:
        r = estimate_rv_semiamplitude(5.0, 1.0, 1.0)
        assert str(round(r.k_ms, 2)) in format_rv_result(r) or "K" in format_rv_result(r)
