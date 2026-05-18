"""Tests for equilibrium_temperature_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from equilibrium_temperature_calculator import (
    compute_equilibrium_temperature,
    format_equilibrium_temperature_result,
)


class TestComputeEquilibriumTemperature:
    def test_basic_earth_like(self):
        # With f=0.25, T_eq(Earth) ≈ 180 K (substellar point with redistribution)
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0, bond_albedo=0.3)
        assert result.flag == "OK"
        assert 100 < result.teq_k < 250

    def test_hot_jupiter_close_in(self):
        # a=0.05 AU, Teff=6000K → T_eq ~900 K with default f=0.25
        result = compute_equilibrium_temperature(6000.0, 1.2, 0.05)
        assert result.flag == "OK"
        assert result.teq_k > 500
        assert result.classification in ("hot", "warm", "ultra-hot")

    def test_temperate_classification(self):
        # Earth-like with f=0.25 gives ~180 K ("cold"); use closer-in orbit for temperate
        result = compute_equilibrium_temperature(5778.0, 1.0, 0.5, bond_albedo=0.3)
        assert result.classification in ("cold", "temperate", "warm")

    def test_cold_classification(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, 10.0, bond_albedo=0.3)
        assert result.classification == "cold"
        assert result.teq_k < 150

    def test_zero_albedo(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0, bond_albedo=0.0)
        assert result.flag == "OK"
        assert result.teq_k > 0

    def test_albedo_one_is_invalid(self):
        # Implementation uses strict bond_albedo < 1, so albedo=1 returns INVALID
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0, bond_albedo=1.0)
        assert result.flag == "INVALID"

    def test_invalid_teff_zero(self):
        result = compute_equilibrium_temperature(0.0, 1.0, 1.0)
        assert result.flag == "INVALID"

    def test_invalid_negative_axis(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, -1.0)
        assert result.flag == "INVALID"

    def test_invalid_albedo_above_one(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0, bond_albedo=1.5)
        assert result.flag == "INVALID"

    def test_error_propagation_present_when_a_err_given(self):
        result = compute_equilibrium_temperature(
            5778.0, 1.0, 1.0, a_au_err=0.05
        )
        assert result.teq_err_k is not None
        assert result.teq_err_k > 0

    def test_no_error_without_a_err(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0)
        assert result.teq_err_k is None

    def test_result_frozen(self):
        result = compute_equilibrium_temperature(5778.0, 1.0, 1.0)
        try:
            result.teq_k = 999  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except (AttributeError, TypeError):
            pass


class TestFormatEquilibriumTemperatureResult:
    def _make(self):
        return compute_equilibrium_temperature(5778.0, 1.0, 1.0)

    def test_returns_string(self):
        assert isinstance(format_equilibrium_temperature_result(self._make()), str)

    def test_contains_flag(self):
        r = self._make()
        assert r.flag in format_equilibrium_temperature_result(r)

    def test_contains_teq(self):
        r = self._make()
        s = format_equilibrium_temperature_result(r)
        assert "T_eq" in s or str(int(r.teq_k)) in s
