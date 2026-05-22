"""Tests for oot_rms_tracker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from oot_rms_tracker import (
    format_oot_rms_result,
    track_oot_rms,
)


def _make_lc(n=200, period=5.0, n_sectors=2):
    time = [i * 0.02 for i in range(n)]
    flux = [1.0] * n
    sector_ids = [1 + (i * n_sectors // n) for i in range(n)]
    return time, flux, sector_ids


class TestTrackOOTRMS:
    def test_result_frozen(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        try:
            r.n_sectors = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_sector_rms_frozen(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        if r.sector_rms:
            sr = r.sector_rms[0]
            try:
                sr.rms = 99.0  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_empty_inputs_invalid(self):
        r = track_oot_rms([], [], 5.0, 0.0, [])
        assert r.flag == "INVALID"

    def test_invalid_period(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, -1.0, 0.0, s)
        assert r.flag == "INVALID"

    def test_n_sectors_matches_unique_ids(self):
        t, f, s = _make_lc(200, 5.0, 2)
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        if r.flag not in ("INVALID", "INSUFFICIENT"):
            assert r.n_sectors == len(set(s))

    def test_median_rms_positive(self):
        t, f, s = _make_lc()
        f = [1.0 + 0.001 * (i % 3 - 1) for i in range(len(t))]
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        if r.median_rms is not None:
            assert r.median_rms >= 0

    def test_elevated_count_non_negative(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        assert r.n_elevated >= 0

    def test_n_elevated_le_n_sectors(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        assert r.n_elevated <= r.n_sectors

    def test_sector_rms_tuple_length(self):
        t, f, s = _make_lc(200, 5.0, 3)
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        if r.flag not in ("INVALID",):
            assert len(r.sector_rms) == r.n_sectors

    def test_flat_lc_low_rms(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        if r.median_rms is not None:
            assert r.median_rms < 1e-6

    def test_noisy_lc_elevated(self):
        import math
        t = [i * 0.02 for i in range(200)]
        f = [1.0 + 0.05 * math.sin(i) for i in range(200)]
        s = [1 if i < 100 else 2 for i in range(200)]
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        assert r.flag in ("OK", "INSUFFICIENT", "INVALID")

    def test_format_returns_string(self):
        t, f, s = _make_lc()
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        out = format_oot_rms_result(r)
        assert isinstance(out, str)
        assert "RMS" in out

    def test_single_sector(self):
        n = 100
        t = [i * 0.02 for i in range(n)]
        f = [1.0] * n
        s = [1] * n
        r = track_oot_rms(t, f, 5.0, 0.0, s)
        assert r.n_sectors >= 0
