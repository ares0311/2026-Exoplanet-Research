"""Tests for Skills/flux_periodogram.py."""
import math
import pytest
from Skills.flux_periodogram import (
    PeriodogramResult,
    PeriodogramPeak,
    compute_dft_periodogram,
    find_periodogram_peaks,
    format_periodogram_result,
)


class TestComputeDftPeriodogram:
    def _sine_lc(self, period=5.0, n=200, dt=0.1):
        time = [i * dt for i in range(n)]
        flux = [math.sin(2 * math.pi * t / period) for t in time]
        return time, flux

    def test_returns_result_type(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f)
        assert isinstance(result, PeriodogramResult)

    def test_flag_ok(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f)
        assert result.flag == "OK"

    def test_power_length_matches_freq(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f)
        assert len(result.freq_grid) == len(result.power)

    def test_power_nonnegative(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f)
        assert all(p >= 0 for p in result.power)

    def test_empty_time_invalid(self):
        result = compute_dft_periodogram([], [])
        assert result.flag == "INVALID"

    def test_mismatched_lengths_invalid(self):
        result = compute_dft_periodogram([1.0, 2.0], [1.0])
        assert result.flag == "INVALID"

    def test_custom_n_freqs(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f, n_freqs=100)
        assert len(result.freq_grid) == 100

    def test_peaks_found(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f)
        assert len(result.peaks) > 0

    def test_peak_at_correct_frequency(self):
        t, f = self._sine_lc(period=5.0, n=300, dt=0.1)
        result = compute_dft_periodogram(t, f, freq_min=0.1, freq_max=0.5, n_freqs=500)
        if result.peaks:
            best = result.peaks[0]
            assert abs(best.period_days - 5.0) < 0.5

    def test_with_flux_err(self):
        t, f = self._sine_lc()
        err = [0.01] * len(f)
        result = compute_dft_periodogram(t, f, flux_err=err)
        assert result.flag == "OK"

    def test_freq_grid_within_range(self):
        t, f = self._sine_lc()
        result = compute_dft_periodogram(t, f, freq_min=0.05, freq_max=0.5)
        assert result.freq_grid[0] >= 0.04
        assert result.freq_grid[-1] <= 0.51


class TestFindPeriodogramPeaks:
    def test_returns_list_of_peaks(self):
        t = [i * 0.1 for i in range(200)]
        f = [math.sin(2 * math.pi * ti / 5.0) for ti in t]
        result = compute_dft_periodogram(t, f)
        peaks = find_periodogram_peaks(result, n_peaks=3)
        assert isinstance(peaks, list)

    def test_peak_type(self):
        t = [i * 0.1 for i in range(200)]
        f = [math.sin(2 * math.pi * ti / 5.0) for ti in t]
        result = compute_dft_periodogram(t, f)
        peaks = find_periodogram_peaks(result, n_peaks=3)
        if peaks:
            assert isinstance(peaks[0], PeriodogramPeak)

    def test_peaks_sorted_descending(self):
        t = [i * 0.1 for i in range(200)]
        f = [math.sin(2 * math.pi * ti / 5.0) for ti in t]
        result = compute_dft_periodogram(t, f)
        peaks = find_periodogram_peaks(result, n_peaks=5)
        for i in range(len(peaks) - 1):
            assert peaks[i].power >= peaks[i + 1].power

    def test_empty_periodogram_no_peaks(self):
        result = compute_dft_periodogram([], [])
        peaks = find_periodogram_peaks(result)
        assert peaks == []


class TestFormatPeriodogramResult:
    def test_returns_string(self):
        t = [i * 0.1 for i in range(50)]
        f = [1.0] * 50
        result = compute_dft_periodogram(t, f)
        s = format_periodogram_result(result)
        assert isinstance(s, str)

    def test_contains_flag(self):
        t = [i * 0.1 for i in range(50)]
        f = [1.0] * 50
        result = compute_dft_periodogram(t, f)
        s = format_periodogram_result(result)
        assert "Flag" in s or "flag" in s.lower()
