"""Tests for Skills/rv_orbital_solution_sampler.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from rv_orbital_solution_sampler import (
    compute_rv_orbital_solution,
    format_rv_orbital_solution_result,
)


class TestComputeRvOrbitalSolution:
    def test_ok_flag(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0)
        assert r.flag == "OK"

    def test_planet_mass_sini_positive(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0)
        assert r.planet_mass_sini_mearth > 0.0

    def test_larger_k1_larger_mass(self) -> None:
        r1 = compute_rv_orbital_solution(10.0, 10.0)
        r2 = compute_rv_orbital_solution(100.0, 10.0)
        assert r2.planet_mass_sini_mearth > r1.planet_mass_sini_mearth

    def test_longer_period_larger_mass(self) -> None:
        r_short = compute_rv_orbital_solution(10.0, 3.0)
        r_long = compute_rv_orbital_solution(10.0, 365.0)
        assert r_long.planet_mass_sini_mearth > r_short.planet_mass_sini_mearth

    def test_k1_preserved(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0)
        assert r.k1_ms == 10.0

    def test_period_preserved(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0)
        assert r.period_days == 10.0

    def test_eccentricity_preserved(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0, eccentricity=0.3)
        assert r.eccentricity == 0.3

    def test_omega_preserved(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0, omega_deg=45.0)
        assert r.omega_deg == 45.0

    def test_invalid_k1(self) -> None:
        r = compute_rv_orbital_solution(0.0, 10.0)
        assert r.flag == "INVALID_K1"

    def test_invalid_period(self) -> None:
        r = compute_rv_orbital_solution(10.0, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_stellar_mass(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0, stellar_mass_msun=0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_invalid_eccentricity(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_format_returns_string(self) -> None:
        r = compute_rv_orbital_solution(10.0, 10.0)
        s = format_rv_orbital_solution_result(r)
        assert isinstance(s, str)
        assert r.flag in s
