"""Tests for Skills/transit_baseline_flux_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_baseline_flux_estimator import estimate_baseline_flux, format_baseline_flux_result


class TestEstimateBaselineFlux:
    def test_basic_ok(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0)
        assert r.flag == "OK"

    def test_flux_positive(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0)
        assert r.flux_photons_per_s > 0

    def test_total_counts_positive(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0)
        assert r.total_counts > 0

    def test_shot_noise_decreases_with_brighter_star(self) -> None:
        r_bright = estimate_baseline_flux(8.0, 1e6, 100.0)
        r_faint = estimate_baseline_flux(12.0, 1e6, 100.0)
        assert r_bright.shot_noise_ppm < r_faint.shot_noise_ppm

    def test_invalid_zero_point(self) -> None:
        r = estimate_baseline_flux(10.0, 0.0, 100.0)
        assert r.flag == "INVALID_ZERO_POINT"

    def test_invalid_aperture(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 0.0)
        assert r.flag == "INVALID_APERTURE"

    def test_invalid_throughput_zero(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0, throughput=0.0)
        assert r.flag == "INVALID_THROUGHPUT"

    def test_invalid_throughput_gt1(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0, throughput=1.5)
        assert r.flag == "INVALID_THROUGHPUT"

    def test_invalid_exposure_time(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0, exposure_time_s=0.0)
        assert r.flag == "INVALID_EXPOSURE_TIME"

    def test_magnitude_stored(self) -> None:
        r = estimate_baseline_flux(12.5, 1e6, 100.0)
        assert r.magnitude == 12.5

    def test_flux_scales_with_aperture(self) -> None:
        r1 = estimate_baseline_flux(10.0, 1e6, 100.0)
        r2 = estimate_baseline_flux(10.0, 1e6, 200.0)
        assert abs(r2.flux_photons_per_s / r1.flux_photons_per_s - 2.0) < 1e-9

    def test_format_returns_string(self) -> None:
        r = estimate_baseline_flux(10.0, 1e6, 100.0)
        s = format_baseline_flux_result(r)
        assert isinstance(s, str)
        assert "Flux" in s
