"""Tests for Skills/stellar_wind_photoionization_balance.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_wind_photoionization_balance import (
    WindPhotoionizationResult,
    estimate_wind_photoionization_balance,
    format_wind_photoionization_result,
)


class TestEstimateWindPhotoionizationBalance:
    def test_returns_result_type(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert isinstance(r, WindPhotoionizationResult)

    def test_flag_ok(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.flag == "OK"

    def test_wind_ram_pressure_positive(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.wind_ram_pressure_pa > 0.0

    def test_xuv_flux_positive(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.xuv_flux_ergcm2s > 0.0

    def test_ionization_timescale_positive(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.ionization_timescale_s > 0.0

    def test_mass_loss_rate_positive(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.mass_loss_rate_gs > 0.0

    def test_magnetosphere_standoff_positive(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.magnetosphere_standoff_rp > 0.0

    def test_atmosphere_class_string(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.atmosphere_class in ("RETAINED", "VULNERABLE", "STRIPPED", "UNKNOWN")

    def test_closer_orbit_higher_xuv(self):
        r_close = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=0.1, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        r_far = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=5.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r_close.xuv_flux_ergcm2s > r_far.xuv_flux_ergcm2s

    def test_younger_star_higher_xuv(self):
        r_young = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=0.5,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        r_old = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=8.0,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r_young.xuv_flux_ergcm2s > r_old.xuv_flux_ergcm2s

    def test_invalid_stellar_mass(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=0.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.flag != "OK"
        assert math.isnan(r.wind_ram_pressure_pa)

    def test_invalid_distance(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=0.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.flag != "OK"

    def test_invalid_age(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=0.0,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        assert r.flag != "OK"

    def test_frozen_dataclass(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        try:
            r.atmosphere_class = "RETAINED"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatWindPhotoionizationResult:
    def test_ok_returns_table(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=1.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        out = format_wind_photoionization_result(r)
        assert "XUV flux" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = estimate_wind_photoionization_balance(
            stellar_mass_msun=0.0, stellar_radius_rsun=1.0, stellar_age_gyr=4.6,
            orbital_distance_au=1.0, planet_mass_mearth=1.0, planet_radius_rearth=1.0
        )
        out = format_wind_photoionization_result(r)
        assert "flag=" in out
