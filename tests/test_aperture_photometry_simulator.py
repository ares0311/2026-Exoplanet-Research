"""Tests for Skills/aperture_photometry_simulator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from aperture_photometry_simulator import format_aperture_sim_result, simulate_aperture_photometry


class TestSimulateAperturePhotometry:
    def test_basic_ok(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 3.0)
        assert r.flag == "OK"

    def test_snr_positive(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 3.0)
        assert r.snr > 0

    def test_invalid_target_counts(self) -> None:
        r = simulate_aperture_photometry(0.0, 5.0, 3.0)
        assert r.flag == "INVALID_TARGET_COUNTS"

    def test_invalid_aperture(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 0.0)
        assert r.flag == "INVALID_APERTURE"

    def test_invalid_n_exposures(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 3.0, n_exposures=0)
        assert r.flag == "INVALID_N_EXPOSURES"

    def test_snr_increases_with_target(self) -> None:
        r1 = simulate_aperture_photometry(1000.0, 1.0, 2.0)
        r2 = simulate_aperture_photometry(10000.0, 1.0, 2.0)
        assert r2.snr > r1.snr

    def test_snr_increases_with_n_exposures(self) -> None:
        r1 = simulate_aperture_photometry(1000.0, 1.0, 2.0, n_exposures=1)
        r2 = simulate_aperture_photometry(1000.0, 1.0, 2.0, n_exposures=10)
        assert r2.snr > r1.snr

    def test_sky_zero_no_sky_noise(self) -> None:
        r = simulate_aperture_photometry(10000.0, 0.0, 2.0, read_noise_electrons=0.0)
        # noise should be sqrt(signal) only
        import math
        expected_snr = math.sqrt(10000.0)
        assert abs(r.snr - expected_snr) < 1e-6

    def test_aperture_pixels_computed(self) -> None:
        import math
        r = simulate_aperture_photometry(10000.0, 1.0, 2.0)
        assert abs(r.aperture_pixels - math.pi * 4.0) < 1e-6

    def test_sky_total_positive(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 3.0)
        assert r.sky_total_counts > 0

    def test_format_returns_string(self) -> None:
        r = simulate_aperture_photometry(10000.0, 5.0, 3.0)
        s = format_aperture_sim_result(r)
        assert isinstance(s, str)
        assert "SNR" in s

    def test_negative_target_invalid(self) -> None:
        r = simulate_aperture_photometry(-100.0, 1.0, 2.0)
        assert r.flag == "INVALID_TARGET_COUNTS"
