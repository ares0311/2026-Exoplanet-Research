"""Tests for depth_snr_per_sector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from depth_snr_per_sector import (
    DepthSNRResult,
    compute_depth_snr_per_sector,
    format_depth_snr_result,
)


def _make_lc_sectors(n_sectors=3, n_per=200, period=5.0, depth=0.005):
    time, flux, sids = [], [], []
    for sec in range(n_sectors):
        t0 = sec * n_per * 0.02
        for i in range(n_per):
            t = t0 + i * 0.02
            ph = (t % period) / period
            if ph >= 0.5:
                ph -= 1.0
            f = 1.0 - depth if abs(ph) < 0.04 else 1.0
            time.append(t)
            flux.append(f)
            sids.append(sec + 1)
    return time, flux, sids


class TestComputeDepthSNRPerSector:
    def test_invalid_empty(self):
        r = compute_depth_snr_per_sector([], [], 5.0, 0.0, [])
        assert r.flag == "INVALID"

    def test_invalid_period_zero(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 0.0, 0.0, sids)
        assert r.flag == "INVALID"

    def test_returns_result(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        assert isinstance(r, DepthSNRResult)

    def test_n_sectors_correct(self):
        time, flux, sids = _make_lc_sectors(n_sectors=3)
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert r.n_sectors == 3

    def test_sector_snrs_length(self):
        time, flux, sids = _make_lc_sectors(n_sectors=2)
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert len(r.sector_snrs) == r.n_sectors

    def test_depth_positive_for_transit(self):
        time, flux, sids = _make_lc_sectors(depth=0.01)
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids, duration_hours=1.5)
        if r.flag == "OK":
            for s in r.sector_snrs:
                assert s.depth_ppm >= 0 or s.n_in_transit < 2

    def test_with_flux_err(self):
        time, flux, sids = _make_lc_sectors()
        errs = [0.0005] * len(flux)
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids, flux_err=errs)
        assert r.flag in ("OK", "INSUFFICIENT")

    def test_mean_depth_present_on_ok(self):
        time, flux, sids = _make_lc_sectors(depth=0.01)
        errs = [0.0002] * len(flux)
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids,
                                         flux_err=errs, duration_hours=1.5)
        if r.flag == "OK" and r.mean_depth_ppm is not None:
            assert r.mean_depth_ppm >= 0

    def test_result_frozen(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        try:
            r.n_sectors = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_sector_snr_frozen(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        if r.sector_snrs:
            s = r.sector_snrs[0]
            try:
                s.depth_ppm = 0.0  # type: ignore[misc]
                raise AssertionError()
            except (AttributeError, TypeError):
                pass


class TestFormatDepthSNRResult:
    def test_returns_string(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        assert isinstance(format_depth_snr_result(r), str)

    def test_contains_flag(self):
        time, flux, sids = _make_lc_sectors()
        r = compute_depth_snr_per_sector(time, flux, 5.0, 0.0, sids)
        assert r.flag in format_depth_snr_result(r)
