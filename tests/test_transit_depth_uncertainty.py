"""Tests for Skills/transit_depth_uncertainty.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_depth_uncertainty import compute_depth_uncertainty, format_depth_uncertainty


class TestTransitDepthUncertainty:
    def test_basic_ok(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0)
        assert r.flag == "OK"
        assert r.depth_err_total_ppm > 0.0

    def test_snr_positive(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0)
        assert r.snr > 0.0

    def test_deeper_transit_higher_snr(self) -> None:
        r1 = compute_depth_uncertainty(500.0, 50, 500, 200.0)
        r2 = compute_depth_uncertainty(2000.0, 50, 500, 200.0)
        assert r2.snr > r1.snr

    def test_more_in_transit_lower_error(self) -> None:
        r1 = compute_depth_uncertainty(1000.0, 10, 500, 200.0)
        r2 = compute_depth_uncertainty(1000.0, 100, 500, 200.0)
        assert r2.depth_err_photon_ppm < r1.depth_err_photon_ppm

    def test_systematic_floor_adds_in_quadrature(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0, systematic_floor_ppm=100.0)
        assert r.depth_err_systematic_ppm == 100.0
        assert r.depth_err_total_ppm > r.depth_err_photon_ppm

    def test_invalid_depth_negative(self) -> None:
        r = compute_depth_uncertainty(-100.0, 50, 500, 200.0)
        assert r.flag == "INVALID_DEPTH"
        assert math.isnan(r.depth_err_total_ppm)

    def test_invalid_n_in_transit(self) -> None:
        r = compute_depth_uncertainty(1000.0, 0, 500, 200.0)
        assert r.flag == "INVALID_N_IN_TRANSIT"

    def test_invalid_n_out_transit(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 0, 200.0)
        assert r.flag == "INVALID_N_OUT_TRANSIT"

    def test_invalid_rms(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 0.0)
        assert r.flag == "INVALID_RMS"

    def test_photon_err_stored(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0)
        assert r.depth_err_photon_ppm > 0.0

    def test_total_gte_photon(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0)
        assert r.depth_err_total_ppm >= r.depth_err_photon_ppm

    def test_format_returns_string(self) -> None:
        r = compute_depth_uncertainty(1000.0, 50, 500, 200.0)
        s = format_depth_uncertainty(r)
        assert isinstance(s, str)
        assert "SNR" in s
