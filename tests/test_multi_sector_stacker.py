"""Tests for Skills.multi_sector_stacker."""
from __future__ import annotations

import pytest
from Skills.multi_sector_stacker import (
    StackedLC,
    format_stack_summary,
    stack_sectors,
)


def _sector(n: int = 50, sector: int = 1, offset: float = 0.0):
    dt = 2.0 / 1440.0
    time = [i * dt + offset for i in range(n)]
    flux = [1.0] * n
    return {"time": time, "flux": flux, "sector": sector}


class TestStackSectors:
    def test_returns_stacked_lc(self) -> None:
        s = stack_sectors([_sector()])
        assert isinstance(s, StackedLC)

    def test_empty_input(self) -> None:
        s = stack_sectors([])
        assert len(s.time) == 0

    def test_empty_sector(self) -> None:
        s = stack_sectors([{"time": [], "flux": [], "sector": 1}])
        assert len(s.time) == 0

    def test_n_cadences_raw_tracked(self) -> None:
        s = stack_sectors([_sector(50)])
        assert s.n_cadences_raw == 50

    def test_two_sectors_combined(self) -> None:
        s1 = _sector(50, sector=1, offset=0.0)
        s2 = _sector(50, sector=2, offset=1.0)
        stacked = stack_sectors([s1, s2])
        assert len(stacked.time) > 50

    def test_time_sorted(self) -> None:
        s1 = _sector(30, sector=1, offset=2.0)
        s2 = _sector(30, sector=2, offset=0.0)
        stacked = stack_sectors([s1, s2])
        assert list(stacked.time) == sorted(stacked.time)

    def test_outlier_clipped(self) -> None:
        s = _sector(50)
        s["flux"][25] = 100.0  # huge outlier
        stacked = stack_sectors([s], sigma_clip=3.0)
        assert stacked.n_cadences_clipped >= 1

    def test_no_normalize_preserves_scale(self) -> None:
        s = _sector(50)
        s["flux"] = [2.0] * 50  # all 2.0
        stacked = stack_sectors([s], normalize=False)
        if stacked.flux:
            assert abs(stacked.flux[0] - 2.0) < 1e-9

    def test_sector_ids_tracked(self) -> None:
        s1 = _sector(10, sector=3)
        stacked = stack_sectors([s1])
        assert all(sid == 3 for sid in stacked.sector_ids)

    def test_flux_err_defaulted_to_zero(self) -> None:
        s = _sector(10)
        stacked = stack_sectors([s])
        assert all(e == pytest.approx(0.0) for e in stacked.flux_err)


class TestFormatStackSummary:
    def test_returns_string(self) -> None:
        s = stack_sectors([_sector()])
        assert isinstance(format_stack_summary(s), str)

    def test_contains_sectors_combined(self) -> None:
        s = stack_sectors([_sector()])
        assert "Sectors" in format_stack_summary(s)
