"""Tests for Skills/radial_velocity_window_planner.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from radial_velocity_window_planner import format_rv_plan, plan_rv_windows


class TestRVWindowPlanner:
    def test_basic_windows(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=4)
        assert r.flag == "OK"
        assert r.n_windows == 4

    def test_invalid_period(self) -> None:
        r = plan_rv_windows(0.0, 2459000.0, 2459000.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_epoch(self) -> None:
        import math
        r = plan_rv_windows(5.0, math.nan, 2459000.0)
        assert r.flag == "INVALID_EPOCH"

    def test_invalid_n_windows(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=0)
        assert r.flag == "INVALID_N_WINDOWS"

    def test_windows_after_start_bjd(self) -> None:
        start = 2459100.0
        r = plan_rv_windows(5.0, 2459000.0, start, n_windows=4)
        for w in r.windows:
            assert w.bjd >= start

    def test_quadrature_phases(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=4)
        phases = {w.phase for w in r.windows}
        assert 0.25 in phases or 0.75 in phases

    def test_rv_fraction_at_quadrature(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=4)
        for w in r.windows:
            if w.phase in (0.25, 0.75):
                assert abs(w.rv_fraction - 1.0) < 0.01

    def test_window_labels(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=4)
        labels = {w.label for w in r.windows}
        assert "max_blueshift" in labels or "max_redshift" in labels

    def test_include_transit(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=8, include_transit=True)
        labels = {w.label for w in r.windows}
        assert "transit" in labels

    def test_window_count_matches(self) -> None:
        r = plan_rv_windows(3.0, 2459000.0, 2459000.0, n_windows=6)
        assert len(r.windows) == 6

    def test_bjds_increasing(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=4)
        bjds = [w.bjd for w in r.windows]
        assert bjds == sorted(bjds)

    def test_format_returns_string(self) -> None:
        r = plan_rv_windows(5.0, 2459000.0, 2459000.0, n_windows=2)
        s = format_rv_plan(r)
        assert isinstance(s, str)
        assert "RV" in s
