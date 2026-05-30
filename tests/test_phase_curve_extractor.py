"""Tests for Skills/phase_curve_extractor.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from phase_curve_extractor import PhaseCurveResult, extract_phase_curve, format_phase_curve_result


class TestPhaseCurveResult:
    def test_dataclass_fields(self):
        r = PhaseCurveResult(
            amplitude_ppm=100.0, phase_offset_rad=0.0,
            baseline_flux=1.0, n_bins=50, flag="OK"
        )
        assert r.amplitude_ppm == 100.0
        assert r.flag == "OK"

    def test_frozen(self):
        r = PhaseCurveResult(amplitude_ppm=100.0, phase_offset_rad=0.0,
                             baseline_flux=1.0, n_bins=50)
        try:
            r.amplitude_ppm = 0
            raise AssertionError()
        except Exception:
            pass


class TestExtractPhaseCurve:
    def _make_sinusoidal(self, n=200, period=10.0, epoch=0.0, amp=0.001):
        time = [i * 0.1 for i in range(n)]
        flux = [1.0 + amp * math.cos(2 * math.pi * t / period) for t in time]
        return time, flux

    def test_basic_extraction(self):
        time, flux = self._make_sinusoidal()
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=50)
        assert r.flag == "OK"
        assert r.amplitude_ppm > 0

    def test_flat_flux_low_amplitude(self):
        time = list(range(100))
        flux = [1.0] * 100
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=20)
        assert r.flag == "OK"
        assert r.amplitude_ppm < 1.0  # essentially zero

    def test_error_on_empty_input(self):
        r = extract_phase_curve([], [], 10.0, 0.0)
        assert r.flag == "ERROR"

    def test_error_on_single_point(self):
        r = extract_phase_curve([0.0], [1.0], 10.0, 0.0)
        assert r.flag == "ERROR"

    def test_error_on_zero_period(self):
        time = list(range(10))
        flux = [1.0] * 10
        r = extract_phase_curve(time, flux, 0.0, 0.0)
        assert r.flag == "ERROR"

    def test_error_on_mismatched_arrays(self):
        r = extract_phase_curve([0.0, 1.0], [1.0], 10.0, 0.0)
        assert r.flag == "ERROR"

    def test_n_bins_used(self):
        time, flux = self._make_sinusoidal()
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=20)
        assert r.n_bins <= 20

    def test_baseline_near_one(self):
        time, flux = self._make_sinusoidal(amp=0.001)
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=50)
        assert abs(r.baseline_flux - 1.0) < 0.01

    def test_phase_offset_range(self):
        time, flux = self._make_sinusoidal()
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=50)
        assert -math.pi <= r.phase_offset_rad <= math.pi


class TestFormatPhaseCurve:
    def test_returns_string(self):
        time = list(range(100))
        flux = [1.0] * 100
        r = extract_phase_curve(time, flux, 10.0, 0.0, n_bins=20)
        s = format_phase_curve_result(r)
        assert isinstance(s, str)

    def test_contains_amplitude(self):
        time = list(range(100))
        flux = [1.0] * 100
        r = extract_phase_curve(time, flux, 10.0, 0.0)
        s = format_phase_curve_result(r)
        assert "ppm" in s
