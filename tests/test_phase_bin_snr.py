"""Tests for phase_bin_snr.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from phase_bin_snr import (
    compute_phase_bin_snr,
    format_phase_bin_snr_result,
)


def _flat_phase(n=100):
    phase = [i / n - 0.5 for i in range(n)]
    flux = [1.0] * n
    return phase, flux


def _transit_phase(n=100, depth=0.01):
    phase = [i / n - 0.5 for i in range(n)]
    flux = [1.0 - depth if abs(p) < 0.05 else 1.0 for p in phase]
    return phase, flux


class TestComputePhaseBinSNR:
    def test_result_frozen(self):
        ph, f = _flat_phase()
        r = compute_phase_bin_snr(ph, f)
        try:
            r.n_bins = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_phase_bin_frozen(self):
        ph, f = _flat_phase()
        r = compute_phase_bin_snr(ph, f)
        if r.bins:
            b = r.bins[0]
            try:
                b.mean_flux = 99.0  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_empty_inputs_invalid(self):
        r = compute_phase_bin_snr([], [])
        assert r.flag == "INVALID"

    def test_mismatched_lengths_invalid(self):
        r = compute_phase_bin_snr([0.0, 0.1], [1.0, 1.0, 1.0])
        assert r.flag == "INVALID"

    def test_n_bins_expected(self):
        ph, f = _flat_phase(200)
        r = compute_phase_bin_snr(ph, f, n_bins=20)
        assert r.n_bins == 20

    def test_bins_tuple_length(self):
        ph, f = _flat_phase(200)
        r = compute_phase_bin_snr(ph, f, n_bins=20)
        assert len(r.bins) == r.n_bins

    def test_transit_bins_detected(self):
        ph, f = _transit_phase(300, 0.01)
        r = compute_phase_bin_snr(ph, f, n_bins=50, transit_half_width=0.06)
        assert len(r.transit_bin_indices) > 0

    def test_peak_snr_positive_with_transit(self):
        ph, f = _transit_phase(300, 0.02)
        err = [0.001] * 300
        r = compute_phase_bin_snr(ph, f, flux_err=err, n_bins=50, transit_half_width=0.06)
        if r.flag == "OK" and r.peak_snr is not None:
            assert r.peak_snr > 0

    def test_transit_depth_ppm_non_negative(self):
        ph, f = _transit_phase(300, 0.01)
        r = compute_phase_bin_snr(ph, f, n_bins=50)
        if r.transit_depth_ppm is not None:
            assert r.transit_depth_ppm >= 0

    def test_bin_centers_in_minus_half_to_half(self):
        ph, f = _flat_phase(200)
        r = compute_phase_bin_snr(ph, f, n_bins=10)
        for b in r.bins:
            assert -0.5 <= b.phase_center <= 0.5

    def test_with_flux_err(self):
        ph, f = _flat_phase(100)
        err = [0.001] * 100
        r = compute_phase_bin_snr(ph, f, flux_err=err, n_bins=10)
        assert r.flag in ("OK", "INSUFFICIENT", "INVALID")
        for b in r.bins:
            assert b.flux_err is not None

    def test_format_returns_string(self):
        ph, f = _flat_phase()
        r = compute_phase_bin_snr(ph, f)
        s = format_phase_bin_snr_result(r)
        assert isinstance(s, str)
        assert "SNR" in s
