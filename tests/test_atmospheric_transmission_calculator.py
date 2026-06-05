"""Tests for Skills/atmospheric_transmission_calculator.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from atmospheric_transmission_calculator import (
    TransmissionResult,
    compute_transmission_amplitude,
    format_transmission_result,
)


class TestComputeTransmissionAmplitude:
    def test_returns_result_type(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert isinstance(r, TransmissionResult)

    def test_flag_ok(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.flag == "OK"

    def test_scale_height_positive(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.scale_height_km > 0.0

    def test_feature_amplitude_positive(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.feature_amplitude_ppm > 0.0

    def test_detectability_string(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.detectability in ("DETECTABLE", "MARGINAL", "UNDETECTABLE")

    def test_hotter_planet_larger_scale_height(self):
        r1 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=1500.0
        )
        r2 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=300.0
        )
        assert r1.scale_height_km > r2.scale_height_km

    def test_lighter_atmosphere_larger_scale_height(self):
        r1 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0,
            mean_molecular_weight_amu=2.3
        )
        r2 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0,
            mean_molecular_weight_amu=28.0
        )
        assert r1.scale_height_km > r2.scale_height_km

    def test_invalid_planet_radius(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=0.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.flag != "OK"
        assert math.isnan(r.scale_height_km)

    def test_invalid_planet_mass(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=0.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.flag != "OK"

    def test_invalid_temperature(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=0.0
        )
        assert r.flag != "OK"

    def test_n_scale_heights_affects_amplitude(self):
        r1 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0,
            n_scale_heights=10.0
        )
        r2 = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0,
            n_scale_heights=2.0
        )
        assert r1.feature_amplitude_ppm > r2.feature_amplitude_ppm

    def test_frozen_dataclass(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        try:
            r.scale_height_km = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_signal_per_scale_height_positive(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        assert r.signal_per_scale_height_ppm > 0.0


class TestFormatTransmissionResult:
    def test_ok_returns_table(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=2.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        out = format_transmission_result(r)
        assert "Scale height" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = compute_transmission_amplitude(
            planet_radius_rearth=0.0, planet_mass_mearth=5.0,
            stellar_radius_rsun=1.0, equilibrium_temperature_k=700.0
        )
        out = format_transmission_result(r)
        assert "flag=" in out
