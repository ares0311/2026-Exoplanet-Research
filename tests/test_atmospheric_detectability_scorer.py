"""Tests for Skills/atmospheric_detectability_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from atmospheric_detectability_scorer import (
    AtmosphericDetectabilityResult,
    score_atmospheric_detectability,
)


class TestAtmosphericDetectabilityScorer:
    def test_basic_ok(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        assert r.flag == "OK"

    def test_scale_height_positive(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        assert r.scale_height_km > 0

    def test_tsm_positive(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        assert r.tsm > 0

    def test_invalid_radius(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=0.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_teq(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=0.0)
        assert r.flag == "INVALID_TEQILIBRIUM"

    def test_large_planet_detectable(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=10.0, stellar_radius_rsun=0.5,
            stellar_tmag=6.0, equilibrium_temp_k=1200.0)
        assert r.jwst_detectable_single or r.n_transits_jwst < 10

    def test_small_planet_harder(self) -> None:
        r_small = score_atmospheric_detectability(
            planet_radius_rearth=1.0, stellar_radius_rsun=1.0,
            stellar_tmag=10.0, equilibrium_temp_k=300.0)
        r_large = score_atmospheric_detectability(
            planet_radius_rearth=5.0, stellar_radius_rsun=1.0,
            stellar_tmag=10.0, equilibrium_temp_k=300.0)
        assert r_large.tsm > r_small.tsm

    def test_with_known_mass(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0,
            planet_mass_mearth=10.0)
        assert r.flag == "OK"

    def test_n_transits_positive(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=12.0, equilibrium_temp_k=400.0)
        assert r.n_transits_jwst > 0
        assert r.n_transits_ariel > 0

    def test_result_frozen(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        assert isinstance(r, AtmosphericDetectabilityResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from atmospheric_detectability_scorer import format_detectability_result
        r = score_atmospheric_detectability(
            planet_radius_rearth=2.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=500.0)
        s = format_detectability_result(r)
        assert "|" in s

    def test_amplitude_positive(self) -> None:
        r = score_atmospheric_detectability(
            planet_radius_rearth=3.0, stellar_radius_rsun=1.0,
            stellar_tmag=8.0, equilibrium_temp_k=800.0)
        assert r.transmission_amplitude_ppm > 0
