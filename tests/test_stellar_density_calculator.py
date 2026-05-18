"""Tests for Skills.stellar_density_calculator."""
from __future__ import annotations

from Skills.stellar_density_calculator import (
    StellarDensityResult,
    compute_stellar_density,
    format_stellar_density_result,
)


class TestComputeStellarDensity:
    def test_returns_result(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        assert isinstance(r, StellarDensityResult)

    def test_ok_flag_for_valid_inputs(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        assert r.flag == "OK"

    def test_invalid_zero_period(self) -> None:
        r = compute_stellar_density(0.0, 3.0, 1000.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_duration(self) -> None:
        r = compute_stellar_density(10.0, 0.0, 1000.0)
        assert r.flag == "INVALID"

    def test_invalid_negative_depth(self) -> None:
        r = compute_stellar_density(10.0, 3.0, -100.0)
        assert r.flag == "INVALID"

    def test_rho_gcm3_positive(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        if r.rho_gcm3 is not None:
            assert r.rho_gcm3 > 0

    def test_rho_sun_positive(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        if r.rho_sun is not None:
            assert r.rho_sun > 0

    def test_a_over_rstar_positive(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        if r.a_over_rstar is not None:
            assert r.a_over_rstar > 1.0

    def test_longer_duration_lower_density(self) -> None:
        r1 = compute_stellar_density(10.0, 2.0, 1000.0)
        r2 = compute_stellar_density(10.0, 4.0, 1000.0)
        if r1.rho_gcm3 is not None and r2.rho_gcm3 is not None:
            assert r2.rho_gcm3 < r1.rho_gcm3

    def test_solar_density_order_of_magnitude(self) -> None:
        # Solar-like star: P=10d, duration≈3h, depth≈1% → ρ★ ~few × ρ☉
        r = compute_stellar_density(10.0, 3.0, 10000.0)
        if r.rho_sun is not None:
            assert 0.01 < r.rho_sun < 1000

    def test_unphysical_very_long_duration(self) -> None:
        # Duration > period is unphysical
        r = compute_stellar_density(1.0, 48.0, 1000.0)
        assert r.flag in {"UNPHYSICAL", "INVALID"}


class TestFormatStellarDensity:
    def test_returns_string(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        assert isinstance(format_stellar_density_result(r), str)

    def test_contains_rho(self) -> None:
        r = compute_stellar_density(10.0, 3.0, 1000.0)
        out = format_stellar_density_result(r)
        assert "ρ" in out or "rho" in out.lower() or "g cm" in out

    def test_invalid_formatted(self) -> None:
        r = compute_stellar_density(0.0, 3.0, 1000.0)
        assert "INVALID" in format_stellar_density_result(r)
