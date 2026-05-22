"""Tests for folded_residual_analyzer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from folded_residual_analyzer import (
    FoldedResidualResult,
    analyze_folded_residuals,
    format_residual_result,
)


def _flat_phase(n=100):
    phase = [i / n - 0.5 for i in range(n)]
    flux = [1.0] * n
    return phase, flux


def _gaussian_noise(n=100, sigma=0.001):
    import math
    phase = [i / n - 0.5 for i in range(n)]
    # Deterministic pseudo-noise
    flux = [1.0 + sigma * math.sin(i * 1.234) for i in range(n)]
    return phase, flux


class TestAnalyzeFoldedResiduals:
    def test_invalid_empty(self):
        r = analyze_folded_residuals([], [])
        assert r.flag == "INVALID"

    def test_invalid_mismatched(self):
        r = analyze_folded_residuals([0.1, 0.2], [1.0])
        assert r.flag == "INVALID"

    def test_flat_returns_result(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert isinstance(r, FoldedResidualResult)

    def test_rms_nonneg(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert r.rms_residual >= 0

    def test_mad_nonneg(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert r.mad_residual >= 0

    def test_flat_is_gaussian(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert r.is_gaussian

    def test_skewness_present_when_enough_data(self):
        phase, flux = _gaussian_noise(n=50)
        r = analyze_folded_residuals(phase, flux)
        if r.flag == "OK":
            assert r.skewness is not None

    def test_with_depth_model(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux, depth_ppm=1000.0, half_width_phase=0.05)
        assert r.flag in ("OK", "INSUFFICIENT")

    def test_chi2_with_errors(self):
        phase, flux = _flat_phase(n=50)
        errs = [0.001] * 50
        r = analyze_folded_residuals(phase, flux, flux_err=errs, n_bins=10)
        assert r.chi2_reduced is not None

    def test_result_frozen(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        try:
            r.rms_residual = 999.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_insufficient_tiny_input(self):
        r = analyze_folded_residuals([0.1, 0.2, 0.3], [1.0, 1.0, 1.0])
        assert r.flag in ("INSUFFICIENT", "INVALID")


class TestFormatResidualResult:
    def test_returns_string(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert isinstance(format_residual_result(r), str)

    def test_contains_flag(self):
        phase, flux = _flat_phase()
        r = analyze_folded_residuals(phase, flux)
        assert r.flag in format_residual_result(r)
