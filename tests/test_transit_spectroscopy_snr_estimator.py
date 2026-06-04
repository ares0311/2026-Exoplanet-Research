"""Tests for Skills/transit_spectroscopy_snr_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_spectroscopy_snr_estimator import (
    compute_transmission_spectroscopy_snr,
    format_transmission_spectroscopy_result,
)


class TestComputeTransmissionSpectroscopySnr:
    def test_ok_flag(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        assert r.flag == "OK"

    def test_scale_height_positive(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        assert r.scale_height_km > 0

    def test_signal_per_scale_height_positive(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        assert r.signal_per_scale_height_ppm > 0

    def test_hotter_teq_larger_scale_height(self) -> None:
        r_hot = compute_transmission_spectroscopy_snr(2.5, 1.0, 1500.0)
        r_cool = compute_transmission_spectroscopy_snr(2.5, 1.0, 300.0)
        assert r_hot.scale_height_km > r_cool.scale_height_km

    def test_larger_planet_larger_signal(self) -> None:
        r_large = compute_transmission_spectroscopy_snr(4.0, 1.0, 700.0)
        r_small = compute_transmission_spectroscopy_snr(1.5, 1.0, 700.0)
        assert r_large.signal_per_scale_height_ppm > r_small.signal_per_scale_height_ppm

    def test_n_transits_jwst_positive(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        assert r.n_transits_jwst_5sigma >= 1

    def test_ariel_requires_more_transits_than_jwst(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        assert r.n_transits_ariel_5sigma >= r.n_transits_jwst_5sigma

    def test_mass_based_gravity(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0, planet_mass_mearth=10.0)
        assert r.flag == "OK"
        assert r.scale_height_km > 0

    def test_invalid_radius(self) -> None:
        r = compute_transmission_spectroscopy_snr(0.0, 1.0, 700.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_stellar_radius(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 0.0, 700.0)
        assert r.flag == "INVALID_STELLAR_RADIUS"

    def test_invalid_teq(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 0.0)
        assert r.flag == "INVALID_TEQ"

    def test_result_frozen(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        try:
            r.scale_height_km = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_transmission_spectroscopy_snr(2.5, 1.0, 700.0)
        s = format_transmission_spectroscopy_result(r)
        assert isinstance(s, str)
        assert r.flag in s
