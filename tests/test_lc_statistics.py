"""Tests for Skills.lc_statistics."""
from __future__ import annotations

import math
import pytest
from Skills.lc_statistics import LCStats, compute_lc_stats, format_lc_stats


def _flat_lc(n: int = 100, cadence_days: float = 2.0 / 1440.0):
    time = [i * cadence_days for i in range(n)]
    flux = [1.0] * n
    return time, flux


class TestComputeLCStats:
    def test_returns_lcstats(self) -> None:
        t, f = _flat_lc()
        result = compute_lc_stats(t, f)
        assert isinstance(result, LCStats)

    def test_empty_returns_zero(self) -> None:
        result = compute_lc_stats([], [])
        assert result.n_cadences == 0
        assert result.rms_ppm == 0.0

    def test_n_cadences_correct(self) -> None:
        t, f = _flat_lc(50)
        result = compute_lc_stats(t, f)
        assert result.n_cadences == 50

    def test_flat_lc_low_rms(self) -> None:
        t, f = _flat_lc(200)
        result = compute_lc_stats(t, f)
        assert result.rms_ppm < 1.0

    def test_noisy_lc_higher_rms(self) -> None:
        import random
        rng = random.Random(42)
        t, f = _flat_lc(200)
        f_noisy = [v + rng.gauss(0, 0.001) for v in f]
        result = compute_lc_stats(t, f_noisy)
        assert result.rms_ppm > 100.0

    def test_outlier_count(self) -> None:
        t, f = _flat_lc(100)
        f[50] = 1.1  # strong outlier
        result = compute_lc_stats(t, f)
        assert result.n_outliers >= 1

    def test_coverage_fraction_full(self) -> None:
        t, f = _flat_lc(100)
        result = compute_lc_stats(t, f, cadence_minutes=2.0)
        assert result.coverage_fraction <= 1.0

    def test_median_flux_approx_one(self) -> None:
        t, f = _flat_lc(100)
        result = compute_lc_stats(t, f)
        assert abs(result.median_flux - 1.0) < 1e-6

    def test_cdpp_nonnegative(self) -> None:
        t, f = _flat_lc(200)
        result = compute_lc_stats(t, f, transit_duration_hours=2.0)
        assert result.cdpp_ppm >= 0.0

    def test_flux_err_used_for_photon_noise(self) -> None:
        t, f = _flat_lc(50)
        err = [1e-4] * 50
        result = compute_lc_stats(t, f, flux_err=err)
        assert abs(result.photon_noise_ppm - 100.0) < 10.0

    def test_single_point(self) -> None:
        result = compute_lc_stats([0.0], [1.0])
        assert result.n_cadences == 1


class TestFormatLCStats:
    def test_format_returns_string(self) -> None:
        t, f = _flat_lc()
        stats = compute_lc_stats(t, f)
        text = format_lc_stats(stats)
        assert isinstance(text, str)

    def test_format_contains_cdpp(self) -> None:
        t, f = _flat_lc()
        stats = compute_lc_stats(t, f)
        assert "CDPP" in format_lc_stats(stats)

    def test_format_contains_rms(self) -> None:
        t, f = _flat_lc()
        stats = compute_lc_stats(t, f)
        assert "RMS" in format_lc_stats(stats)
