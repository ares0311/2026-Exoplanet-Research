"""Tests for Skills.detrending_comparator."""
from __future__ import annotations

import numpy as np
from Skills.detrending_comparator import DetrendingReport, compare_detrending


def _lc(n: int = 1000, period: float = 10.0, depth_ppm: float = 2000.0) -> tuple:
    rng = np.random.default_rng(42)
    time = np.linspace(2458000.0, 2458027.0, n)
    flux = np.ones(n) + rng.normal(0, 2e-4, n)
    return time, flux


class TestCompareDetrending:
    def test_recommended_window_is_one_of_input_windows(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51, 101, 201))
        assert report.recommended_window in (51, 101, 201)

    def test_scatter_by_window_has_entry_for_each_window(self) -> None:
        time, flux = _lc()
        windows = (51, 101)
        report = compare_detrending(time, flux, windows=windows)
        assert set(report.scatter_by_window.keys()) == set(windows)

    def test_snr_by_window_has_entry_for_each_window(self) -> None:
        time, flux = _lc()
        windows = (51, 101)
        report = compare_detrending(time, flux, windows=windows)
        assert set(report.signal_snr_by_window.keys()) == set(windows)

    def test_scatter_values_non_negative(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51, 101))
        for s in report.scatter_by_window.values():
            assert s >= 0.0

    def test_snr_values_non_negative(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51, 101))
        for s in report.signal_snr_by_window.values():
            assert s >= 0.0

    def test_reason_mentions_window(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51, 101))
        assert "Window" in report.reason or str(report.recommended_window) in report.reason

    def test_returns_detrending_report_instance(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51,))
        assert isinstance(report, DetrendingReport)

    def test_single_window_that_window_is_recommended(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(101,))
        assert report.recommended_window == 101

    def test_period_days_parameter_accepted(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51,), period_days=5.0)
        assert isinstance(report, DetrendingReport)

    def test_epoch_bjd_defaults_to_time_start(self) -> None:
        time, flux = _lc()
        report1 = compare_detrending(time, flux, windows=(51,))
        report2 = compare_detrending(time, flux, windows=(51,), epoch_bjd=float(time[0]))
        assert report1.recommended_window == report2.recommended_window

    def test_custom_depth_ppm_accepted(self) -> None:
        time, flux = _lc()
        report = compare_detrending(time, flux, windows=(51,), depth_ppm=500.0)
        assert isinstance(report, DetrendingReport)

    def test_flux_err_parameter_accepted(self) -> None:
        time, flux = _lc()
        flux_err = np.full_like(flux, 2e-4)
        report = compare_detrending(time, flux, flux_err, windows=(51,))
        assert isinstance(report, DetrendingReport)
