"""Tests for Skills/ttv_model_fitter.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ttv_model_fitter import fit_ttv_sinusoid, format_ttv_fit_result


class TestTTVModelFitter:
    def _sinusoidal_oc(self, n: int, amp: float = 5.0, period: float = 10.0) -> list[float]:
        return [amp * math.sin(2 * math.pi * i / period) for i in range(n)]

    def test_insufficient_data(self) -> None:
        r = fit_ttv_sinusoid([1.0, 2.0, 3.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_noise(self) -> None:
        r = fit_ttv_sinusoid([0.0] * 6, noise_minutes=0.0)
        assert r.flag == "INVALID_NOISE"

    def test_flat_oc_low_amplitude(self) -> None:
        r = fit_ttv_sinusoid([0.0] * 10)
        assert r.flag == "OK"
        assert r.amplitude_minutes < 0.01

    def test_sinusoidal_oc_detected(self) -> None:
        oc = self._sinusoidal_oc(20, amp=10.0, period=8.0)
        r = fit_ttv_sinusoid(oc, noise_minutes=0.1)
        assert r.flag == "OK"
        assert r.amplitude_minutes > 1.0

    def test_n_points_correct(self) -> None:
        oc = [1.0] * 8
        r = fit_ttv_sinusoid(oc)
        assert r.n_points == 8

    def test_rms_residual_non_negative(self) -> None:
        r = fit_ttv_sinusoid([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5])
        assert r.rms_residual_minutes >= 0.0

    def test_reduced_chi2_positive(self) -> None:
        r = fit_ttv_sinusoid([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
        assert r.reduced_chi2 >= 0.0

    def test_custom_transit_numbers(self) -> None:
        oc = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
        ns = [0, 2, 4, 6, 8, 10, 12, 14]
        r = fit_ttv_sinusoid(oc, transit_numbers=ns)
        assert r.flag == "OK"

    def test_length_mismatch(self) -> None:
        r = fit_ttv_sinusoid([1.0] * 6, transit_numbers=[0, 1, 2])
        assert r.flag == "LENGTH_MISMATCH"

    def test_period_transits_positive(self) -> None:
        oc = self._sinusoidal_oc(12, amp=3.0, period=6.0)
        r = fit_ttv_sinusoid(oc)
        assert r.period_transits > 0.0

    def test_offset_is_float(self) -> None:
        r = fit_ttv_sinusoid([0.5] * 6 + [-0.5] * 6)
        assert isinstance(r.offset_minutes, float)

    def test_format_returns_string(self) -> None:
        r = fit_ttv_sinusoid([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
        s = format_ttv_fit_result(r)
        assert isinstance(s, str)
        assert "amplitude" in s.lower()
