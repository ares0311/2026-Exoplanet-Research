"""Tests for Skills/stellar_wind_ram_pressure.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_wind_ram_pressure import compute_stellar_wind_ram_pressure, format_stellar_wind_result


class TestComputeStellarWindRamPressure:
    def test_ok_flag(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        assert r.flag == "OK"

    def test_ram_pressure_positive(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        assert r.ram_pressure_pa > 0.0

    def test_wind_density_positive(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        assert r.wind_density_kg_m3 > 0.0

    def test_closer_orbit_higher_pressure(self) -> None:
        r_near = compute_stellar_wind_ram_pressure(0.1)
        r_far = compute_stellar_wind_ram_pressure(5.0)
        assert r_near.ram_pressure_pa > r_far.ram_pressure_pa

    def test_standoff_radius_positive(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        assert r.magnetospheric_standoff_rp > 0.0

    def test_stronger_magnetic_larger_standoff(self) -> None:
        r_weak = compute_stellar_wind_ram_pressure(1.0, planet_magnetic_moment_earth=0.1)
        r_strong = compute_stellar_wind_ram_pressure(1.0, planet_magnetic_moment_earth=10.0)
        assert r_strong.magnetospheric_standoff_rp > r_weak.magnetospheric_standoff_rp

    def test_space_weather_class_set(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        assert r.space_weather_class in ("QUIET", "MODERATE", "ACTIVE", "EXTREME")

    def test_high_mdot_active_weather(self) -> None:
        r = compute_stellar_wind_ram_pressure(0.1, stellar_mass_loss_rate_msun_yr=1e-10)
        assert r.space_weather_class in ("ACTIVE", "EXTREME")

    def test_invalid_distance(self) -> None:
        r = compute_stellar_wind_ram_pressure(0.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_invalid_mdot(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0, stellar_mass_loss_rate_msun_yr=0.0)
        assert r.flag == "INVALID_MDOT"

    def test_invalid_velocity(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0, wind_velocity_ms=0.0)
        assert r.flag == "INVALID_VELOCITY"

    def test_result_frozen(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        try:
            r.ram_pressure_pa = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_stellar_wind_ram_pressure(1.0)
        s = format_stellar_wind_result(r)
        assert isinstance(s, str)
        assert r.flag in s
