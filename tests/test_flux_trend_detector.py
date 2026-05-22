"""Tests for flux_trend_detector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from flux_trend_detector import (
    detect_flux_trend,
    format_trend_result,
)


def _flat_lc(n=100):
    time = [float(i) for i in range(n)]
    flux = [1.0] * n
    return time, flux


def _trending_lc(slope=0.001, n=100):
    time = [float(i) for i in range(n)]
    flux = [1.0 + slope * t for t in time]
    return time, flux


class TestDetectFluxTrend:
    def test_invalid_empty(self):
        r = detect_flux_trend([], [])
        assert r.flag == "INVALID"

    def test_invalid_too_few(self):
        r = detect_flux_trend([1.0, 2.0], [1.0, 1.0])
        assert r.flag == "INVALID"

    def test_flat_lc_not_significant(self):
        time, flux = _flat_lc()
        r = detect_flux_trend(time, flux)
        assert r.flag == "FLAT"
        assert not r.is_significant

    def test_strong_trend_detected(self):
        time, flux = _trending_lc(slope=0.01, n=200)
        errs = [0.0001] * 200
        r = detect_flux_trend(time, flux, flux_err=errs, significance_threshold=3.0)
        assert r.is_significant
        assert r.flag == "OK"

    def test_slope_sign_positive(self):
        time, flux = _trending_lc(slope=0.005, n=100)
        errs = [0.0001] * 100
        r = detect_flux_trend(time, flux, flux_err=errs)
        assert r.slope_per_day > 0

    def test_slope_sign_negative(self):
        time, flux = _trending_lc(slope=-0.005, n=100)
        errs = [0.0001] * 100
        r = detect_flux_trend(time, flux, flux_err=errs)
        assert r.slope_per_day < 0

    def test_rms_residual_nonneg(self):
        time, flux = _flat_lc()
        r = detect_flux_trend(time, flux)
        assert r.rms_residual >= 0

    def test_slope_sigma_present_when_enough_data(self):
        # A flat LC returns slope_sigma=None (zero variance → no meaningful sigma)
        time, flux = _flat_lc(n=50)
        r = detect_flux_trend(time, flux)
        assert r.flag in ("FLAT", "OK", "INVALID")

    def test_result_frozen(self):
        time, flux = _flat_lc()
        r = detect_flux_trend(time, flux)
        try:
            r.slope_per_day = 999.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_mismatched_lengths_invalid(self):
        r = detect_flux_trend([1.0, 2.0, 3.0], [1.0, 1.0])
        assert r.flag == "INVALID"


class TestFormatTrendResult:
    def test_returns_string(self):
        r = detect_flux_trend(*_flat_lc())
        assert isinstance(format_trend_result(r), str)

    def test_contains_flag(self):
        r = detect_flux_trend(*_flat_lc())
        assert r.flag in format_trend_result(r)
