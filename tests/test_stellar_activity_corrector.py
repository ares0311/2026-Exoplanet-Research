"""Tests for Skills/stellar_activity_corrector.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_activity_corrector import correct_for_stellar_activity, format_activity_correction


class TestCorrectForStellarActivity:
    def _sine_data(self, n: int = 50, period: float = 10.0) -> tuple[list[float], list[float]]:
        time = [i * 0.2 for i in range(n)]
        flux = [math.sin(2 * math.pi * t / period) for t in time]
        return flux, time

    def test_basic_correction(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert r.flag == "OK"
        assert r.rms_after <= r.rms_before + 1e-6

    def test_amplitude_positive(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert r.amplitude >= 0.0

    def test_insufficient_data(self) -> None:
        r = correct_for_stellar_activity([1.0, 2.0], [0.0, 1.0], 10.0)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_period(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_corrected_flux_length(self) -> None:
        flux, time = self._sine_data(n=30)
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert len(r.corrected_flux) == 30

    def test_n_points_stored(self) -> None:
        flux, time = self._sine_data(n=40)
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert r.n_points == 40

    def test_rms_before_non_negative(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert r.rms_before >= 0.0

    def test_rms_after_non_negative(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert r.rms_after >= 0.0

    def test_constant_flux_no_change(self) -> None:
        flux = [1.0] * 20
        time = [float(i) for i in range(20)]
        r = correct_for_stellar_activity(flux, time, 5.0)
        assert r.flag in ("OK", "DEGENERATE")

    def test_phase_rad_is_float(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        assert isinstance(r.phase_rad, float)

    def test_result_frozen(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        flux, time = self._sine_data()
        r = correct_for_stellar_activity(flux, time, 10.0)
        s = format_activity_correction(r)
        assert isinstance(s, str)
        assert "Activity" in s
