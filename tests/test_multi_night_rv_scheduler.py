"""Tests for Skills/multi_night_rv_scheduler.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_night_rv_scheduler import format_rv_schedule, schedule_rv_observations


class TestScheduleRvObservations:
    def test_basic_schedule(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=10)
        assert r.flag == "OK"
        assert r.n_nights == 10

    def test_phase_coverage_fraction(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=20)
        assert 0.0 <= r.phase_coverage <= 1.0

    def test_high_coverage_many_obs(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=30)
        assert r.phase_coverage > 0.8

    def test_phase_rms_nonneg(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=10)
        assert r.phase_rms >= 0.0

    def test_entries_count_matches(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=8)
        assert len(r.entries) == 8

    def test_phases_in_unit_interval(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=15)
        for e in r.entries:
            assert 0.0 <= e.orbital_phase < 1.0

    def test_invalid_period(self) -> None:
        r = schedule_rv_observations(period_days=0.0, n_observations=5)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_n_observations(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=0)
        assert r.flag == "INVALID_N_OBSERVATIONS"

    def test_t0_bjd_offset(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=5, t0_bjd=2460000.0)
        # All BJDs should start near t0
        for e in r.entries:
            assert e.bjd >= 2460000.0 - 5.0

    def test_baseline_constrained(self) -> None:
        r = schedule_rv_observations(
            period_days=5.0, n_observations=10, baseline_days=100.0
        )
        assert r.flag == "OK"
        assert len(r.entries) == 10

    def test_format_output(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=5)
        s = format_rv_schedule(r)
        assert "|" in s
        assert "phase" in s.lower() or "BJD" in s

    def test_night_index_sequential(self) -> None:
        r = schedule_rv_observations(period_days=5.0, n_observations=5)
        for i, e in enumerate(r.entries):
            assert e.night_index == i
