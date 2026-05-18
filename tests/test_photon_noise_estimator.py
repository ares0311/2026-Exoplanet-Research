"""Tests for Skills.photon_noise_estimator."""
from __future__ import annotations

from Skills.photon_noise_estimator import (
    PhotonNoiseResult,
    estimate_photon_noise,
    format_photon_noise_result,
)


class TestEstimatePhotonNoise:
    def test_returns_result(self) -> None:
        r = estimate_photon_noise(12.0)
        assert isinstance(r, PhotonNoiseResult)

    def test_ok_flag_normal_star(self) -> None:
        r = estimate_photon_noise(12.0)
        assert r.flag == "OK"

    def test_bright_flag(self) -> None:
        r = estimate_photon_noise(4.0)
        assert r.flag == "BRIGHT"

    def test_faint_flag(self) -> None:
        r = estimate_photon_noise(17.0)
        assert r.flag == "FAINT"

    def test_invalid_negative_cadence(self) -> None:
        r = estimate_photon_noise(12.0, cadence_sec=-1.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_pixels(self) -> None:
        r = estimate_photon_noise(12.0, n_pixels=0)
        assert r.flag == "INVALID"

    def test_photon_noise_positive(self) -> None:
        r = estimate_photon_noise(12.0)
        assert r.photon_noise_ppm > 0

    def test_total_noise_geq_photon_noise(self) -> None:
        r = estimate_photon_noise(12.0)
        assert r.total_noise_ppm >= r.photon_noise_ppm

    def test_cdpp_leq_total_noise(self) -> None:
        # CDPP averages over multiple cadences → should be less than per-cadence noise
        r = estimate_photon_noise(12.0, cadence_sec=120.0)
        assert r.cdpp_ppm_hr <= r.total_noise_ppm

    def test_brighter_star_lower_noise(self) -> None:
        r1 = estimate_photon_noise(10.0)
        r2 = estimate_photon_noise(14.0)
        assert r1.photon_noise_ppm < r2.photon_noise_ppm

    def test_systematic_floor_included(self) -> None:
        r1 = estimate_photon_noise(12.0, systematic_floor_ppm=0.0)
        r2 = estimate_photon_noise(12.0, systematic_floor_ppm=100.0)
        assert r2.total_noise_ppm >= r1.total_noise_ppm


class TestFormatPhotonNoise:
    def test_returns_string(self) -> None:
        r = estimate_photon_noise(12.0)
        assert isinstance(format_photon_noise_result(r), str)

    def test_contains_tmag(self) -> None:
        r = estimate_photon_noise(12.0)
        assert "12" in format_photon_noise_result(r)

    def test_contains_flag(self) -> None:
        r = estimate_photon_noise(12.0)
        assert r.flag in format_photon_noise_result(r)
