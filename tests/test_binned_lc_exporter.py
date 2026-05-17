"""Tests for Skills.binned_lc_exporter."""
from __future__ import annotations

from pathlib import Path

import pytest
from Skills.binned_lc_exporter import (
    BinnedLC,
    bin_lightcurve,
    export_binned_lc,
    format_bin_summary,
    load_binned_lc,
)


def _lc(n: int = 100, cadence_minutes: float = 2.0):
    dt = cadence_minutes / 1440.0
    time = [i * dt for i in range(n)]
    flux = [1.0] * n
    return time, flux


class TestBinLightcurve:
    def test_returns_binned_lc(self) -> None:
        t, f = _lc()
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert isinstance(b, BinnedLC)

    def test_empty_input(self) -> None:
        b = bin_lightcurve([], [], bin_minutes=30.0)
        assert len(b.time) == 0

    def test_fewer_bins_than_cadences(self) -> None:
        t, f = _lc(60)  # 60 × 2-min = 2 h → 4 bins at 30-min
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert len(b.time) < 60

    def test_flux_close_to_one(self) -> None:
        t, f = _lc(120)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert all(abs(v - 1.0) < 1e-9 for v in b.flux)

    def test_n_points_per_bin_positive(self) -> None:
        t, f = _lc(60)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert all(n > 0 for n in b.n_points_per_bin)

    def test_cadence_minutes_stored(self) -> None:
        t, f = _lc()
        b = bin_lightcurve(t, f, bin_minutes=15.0)
        assert b.cadence_minutes == pytest.approx(15.0)

    def test_with_flux_err(self) -> None:
        t, f = _lc(60)
        err = [0.001] * 60
        b = bin_lightcurve(t, f, flux_err=err, bin_minutes=30.0)
        assert all(e >= 0.0 for e in b.flux_err)

    def test_time_sorted(self) -> None:
        t, f = _lc(100)
        b = bin_lightcurve(t, f, bin_minutes=10.0)
        assert list(b.time) == sorted(b.time)


class TestExportLoadBinnedLC:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        t, f = _lc(60)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        p = export_binned_lc(b, tmp_path / "binned.json")
        assert p.exists()

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        t, f = _lc(60)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        p = export_binned_lc(b, tmp_path / "binned.json")
        b2 = load_binned_lc(p)
        assert b2.cadence_minutes == pytest.approx(b.cadence_minutes)
        assert len(b2.time) == len(b.time)


class TestFormatBinSummary:
    def test_returns_string(self) -> None:
        t, f = _lc(60)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert isinstance(format_bin_summary(b), str)

    def test_contains_bins(self) -> None:
        t, f = _lc(60)
        b = bin_lightcurve(t, f, bin_minutes=30.0)
        assert "Bins" in format_bin_summary(b)
