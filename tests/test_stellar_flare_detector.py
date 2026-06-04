"""Tests for stellar_flare_detector.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_flare_detector import (
    detect_stellar_flares,
    format_flare_result,
)


def _flat_lc(n=100):
    time = [float(i) * 0.02 for i in range(n)]
    flux = [1.0] * n
    return time, flux


def _flare_lc(n=200, flare_start=80, flare_amp=0.10):
    time = [float(i) * 0.02 for i in range(n)]
    flux = [1.0] * n
    for k in range(flare_start, min(flare_start + 5, n)):
        flux[k] = 1.0 + flare_amp * math.exp(-(k - flare_start) * 0.5)
    return time, flux


class TestDetectStellarFlares:
    def test_invalid_empty_input(self):
        r = detect_stellar_flares([], [])
        assert r.flag == "INVALID"

    def test_invalid_mismatched_lengths(self):
        r = detect_stellar_flares([1.0, 2.0], [1.0, 1.0, 1.0])
        assert r.flag == "INVALID"

    def test_flat_lc_no_flares(self):
        time, flux = _flat_lc()
        r = detect_stellar_flares(time, flux, sigma_threshold=3.0)
        assert r.flag in ("NO_FLARES", "INSUFFICIENT")

    def test_flare_detected(self):
        time, flux = _flare_lc(flare_amp=0.20)
        errs = [0.001] * len(flux)
        r = detect_stellar_flares(time, flux, flux_err=errs,
                                  sigma_threshold=3.0, min_duration_cadences=2)
        assert r.n_flares >= 1
        assert r.flag == "OK"

    def test_n_cadences_correct(self):
        time, flux = _flat_lc(n=150)
        r = detect_stellar_flares(time, flux)
        assert r.n_cadences == 150

    def test_result_frozen(self):
        r = detect_stellar_flares(*_flat_lc())
        try:
            r.n_flares = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_max_amplitude_positive_when_flare(self):
        time, flux = _flare_lc(flare_amp=0.30)
        r = detect_stellar_flares(time, flux, sigma_threshold=2.0, min_duration_cadences=2)
        if r.n_flares > 0:
            assert r.max_amplitude is not None
            assert r.max_amplitude > 0

    def test_energy_proxy_nonnegative(self):
        time, flux = _flare_lc(flare_amp=0.15)
        r = detect_stellar_flares(time, flux, sigma_threshold=2.0)
        if r.flag == "OK":
            assert r.total_flare_energy_proxy >= 0

    def test_flare_indices_within_range(self):
        time, flux = _flare_lc(flare_amp=0.20)
        r = detect_stellar_flares(time, flux, sigma_threshold=2.0)
        for start, end in r.flare_indices:
            assert 0 <= start <= end < len(flux)

    def test_with_flux_err(self):
        time, flux = _flare_lc(flare_amp=0.25)
        errs = [0.002] * len(flux)
        r = detect_stellar_flares(time, flux, flux_err=errs, sigma_threshold=3.0)
        assert r.flag in ("OK", "NO_FLARES", "INSUFFICIENT")

    def test_high_threshold_no_detection(self):
        # Add noise so baseline RMS is non-zero; tiny flare + high sigma => no detection
        import random
        random.seed(42)
        time, flux = _flare_lc(flare_amp=0.05)
        flux = [f + random.gauss(0, 0.01) for f in flux]
        r = detect_stellar_flares(time, flux, sigma_threshold=100.0)
        assert r.n_flares == 0


class TestFormatFlareResult:
    def test_returns_string(self):
        r = detect_stellar_flares(*_flat_lc())
        assert isinstance(format_flare_result(r), str)

    def test_contains_flag(self):
        r = detect_stellar_flares(*_flat_lc())
        assert r.flag in format_flare_result(r)
