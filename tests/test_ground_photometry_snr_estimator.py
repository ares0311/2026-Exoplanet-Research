"""Tests for Skills/ground_photometry_snr_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ground_photometry_snr_estimator import estimate_ground_snr, format_ground_snr


class TestGroundPhotometrySnrEstimator:
    def test_basic_detection(self) -> None:
        r = estimate_ground_snr(10000.0, 10.0)
        assert r.flag == "OK"
        assert r.snr_single > 0.0

    def test_snr_positive(self) -> None:
        r = estimate_ground_snr(5000.0, 11.0, aperture_m=2.0)
        assert r.snr_single > 0.0

    def test_stacked_snr_larger(self) -> None:
        r = estimate_ground_snr(1000.0, 12.0, n_transits=4)
        assert r.snr_stacked > r.snr_single

    def test_stacking_sqrt_n(self) -> None:
        r1 = estimate_ground_snr(1000.0, 12.0, n_transits=1)
        r4 = estimate_ground_snr(1000.0, 12.0, n_transits=4)
        assert abs(r4.snr_stacked / r1.snr_single - 2.0) < 0.01

    def test_brighter_star_higher_snr(self) -> None:
        r_bright = estimate_ground_snr(1000.0, 9.0)
        r_faint = estimate_ground_snr(1000.0, 14.0)
        assert r_bright.snr_single > r_faint.snr_single

    def test_larger_aperture_higher_snr(self) -> None:
        r1 = estimate_ground_snr(1000.0, 12.0, aperture_m=1.0)
        r2 = estimate_ground_snr(1000.0, 12.0, aperture_m=4.0)
        assert r2.snr_single > r1.snr_single

    def test_deeper_transit_higher_snr(self) -> None:
        r1 = estimate_ground_snr(1000.0, 12.0)
        r2 = estimate_ground_snr(5000.0, 12.0)
        assert r2.snr_single > r1.snr_single

    def test_invalid_depth(self) -> None:
        r = estimate_ground_snr(0.0, 12.0)
        assert "INVALID" in r.flag
        assert math.isnan(r.snr_single)

    def test_invalid_tmag(self) -> None:
        r = estimate_ground_snr(1000.0, 0.0)
        assert "INVALID" in r.flag

    def test_invalid_n_transits(self) -> None:
        r = estimate_ground_snr(1000.0, 12.0, n_transits=0)
        assert r.flag == "INVALID_N_TRANSITS"

    def test_noise_ppm_positive(self) -> None:
        r = estimate_ground_snr(1000.0, 12.0)
        assert r.noise_ppm > 0.0

    def test_format_returns_string(self) -> None:
        r = estimate_ground_snr(1000.0, 12.0)
        s = format_ground_snr(r)
        assert isinstance(s, str)
        assert "SNR" in s
