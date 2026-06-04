"""Tests for Skills/rv_precision_requirement.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from rv_precision_requirement import compute_rv_precision_requirement, format_rv_precision_result


class TestComputeRvPrecisionRequirement:
    def test_ok_flag(self) -> None:
        r = compute_rv_precision_requirement(365.0, 317.8)  # Jupiter at 1 yr
        assert r.flag == "OK"

    def test_earth_k_amplitude(self) -> None:
        r = compute_rv_precision_requirement(365.25, 1.0)
        assert abs(r.k_amplitude_ms - 0.09) < 0.02

    def test_hot_jupiter_detectable_harps(self) -> None:
        r = compute_rv_precision_requirement(3.5, 317.8)
        assert r.detectable_harps

    def test_earth_not_detectable_harps(self) -> None:
        r = compute_rv_precision_requirement(365.25, 1.0)
        assert not r.detectable_harps

    def test_earth_not_detectable_pfs(self) -> None:
        r = compute_rv_precision_requirement(365.25, 1.0)
        assert not r.detectable_pfs

    def test_longer_period_smaller_k(self) -> None:
        r_short = compute_rv_precision_requirement(10.0, 10.0)
        r_long = compute_rv_precision_requirement(100.0, 10.0)
        assert r_short.k_amplitude_ms > r_long.k_amplitude_ms

    def test_more_massive_planet_larger_k(self) -> None:
        r1 = compute_rv_precision_requirement(30.0, 10.0)
        r2 = compute_rv_precision_requirement(30.0, 100.0)
        assert r2.k_amplitude_ms > r1.k_amplitude_ms

    def test_invalid_period(self) -> None:
        r = compute_rv_precision_requirement(0.0, 10.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_mass(self) -> None:
        r = compute_rv_precision_requirement(30.0, 0.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_eccentricity(self) -> None:
        r = compute_rv_precision_requirement(30.0, 10.0, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_n_observations_positive(self) -> None:
        r = compute_rv_precision_requirement(30.0, 10.0)
        assert r.n_observations_for_5sigma >= 1

    def test_result_frozen(self) -> None:
        r = compute_rv_precision_requirement(30.0, 10.0)
        try:
            r.k_amplitude_ms = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_rv_precision_requirement(30.0, 10.0)
        s = format_rv_precision_result(r)
        assert isinstance(s, str)
        assert r.flag in s
