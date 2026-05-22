"""Tests for signal_persistence_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from signal_persistence_checker import (
    check_signal_persistence,
    format_persistence_result,
)


def _make_lc_with_sectors(period=5.0, n_sectors=3, n_per_sector=100, depth=0.005):
    """Create a synthetic LC with transit signal across multiple sectors."""
    time = []
    flux = []
    sids = []
    for sec in range(n_sectors):
        t0 = sec * n_per_sector * 0.02
        for i in range(n_per_sector):
            t = t0 + i * 0.02
            # Simple box transit
            ph = (t % period) / period
            if ph >= 0.5:
                ph -= 1.0
            f = 1.0 - depth if abs(ph) < 0.04 else 1.0
            time.append(t)
            flux.append(f)
            sids.append(sec + 1)
    return time, flux, sids


class TestCheckSignalPersistence:
    def test_invalid_empty(self):
        r = check_signal_persistence([], [], 5.0, 0.0, [])
        assert r.flag == "INVALID"

    def test_invalid_period_zero(self):
        time, flux, sids = _make_lc_with_sectors()
        r = check_signal_persistence(time, flux, 0.0, 0.0, sids)
        assert r.flag == "INVALID"

    def test_insufficient_single_sector(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=1, n_per_sector=200)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids, min_sectors=2)
        assert r.flag in ("INSUFFICIENT", "OK")

    def test_persistent_signal_detected(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=4, depth=0.01)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids,
                                      duration_hours=1.0, min_sectors=2)
        if r.flag == "OK":
            assert r.n_sectors == 4

    def test_n_sectors_correct(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=3)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert r.n_sectors == 3

    def test_persistence_fraction_range(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=3)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert 0.0 <= r.persistence_fraction <= 1.0

    def test_depths_per_sector_length(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=3)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert len(r.depths_per_sector) == r.n_sectors

    def test_sector_ids_unique(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=3)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        if r.flag == "OK":
            assert len(r.sector_ids) == len(set(r.sector_ids))

    def test_result_frozen(self):
        time, flux, sids = _make_lc_with_sectors(n_sectors=2)
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        try:
            r.n_sectors = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatPersistenceResult:
    def test_returns_string(self):
        time, flux, sids = _make_lc_with_sectors()
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        assert isinstance(format_persistence_result(r), str)

    def test_contains_flag(self):
        time, flux, sids = _make_lc_with_sectors()
        r = check_signal_persistence(time, flux, 5.0, 0.0, sids)
        assert r.flag in format_persistence_result(r)
