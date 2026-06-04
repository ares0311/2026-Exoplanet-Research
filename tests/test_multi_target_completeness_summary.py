"""Tests for Skills/multi_target_completeness_summary.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_target_completeness_summary import (
    build_completeness_summary,
    format_completeness_summary,
)


class TestBuildCompletenessSummary:
    def _targets(self) -> list[dict]:
        return [
            {"tic_id": "100", "n_injected": 50, "n_recovered": 40,
             "mean_period_days": 5.0, "mean_depth_ppm": 500.0},
            {"tic_id": "200", "n_injected": 30, "n_recovered": 24,
             "mean_period_days": 10.0, "mean_depth_ppm": 300.0},
        ]

    def test_empty_input(self) -> None:
        r = build_completeness_summary([])
        assert r.flag == "NO_RESULTS"
        assert r.n_targets == 0

    def test_two_targets(self) -> None:
        r = build_completeness_summary(self._targets())
        assert r.n_targets == 2
        assert r.flag == "OK"

    def test_total_injected(self) -> None:
        r = build_completeness_summary(self._targets())
        assert r.total_injected == 80

    def test_total_recovered(self) -> None:
        r = build_completeness_summary(self._targets())
        assert r.total_recovered == 64

    def test_mean_recovery_rate(self) -> None:
        r = build_completeness_summary(self._targets())
        assert 0.0 <= r.mean_recovery_rate <= 1.0

    def test_min_max_rate(self) -> None:
        r = build_completeness_summary(self._targets())
        assert r.min_recovery_rate <= r.max_recovery_rate

    def test_entry_recovery_rate(self) -> None:
        r = build_completeness_summary(self._targets())
        for e in r.entries:
            if e.n_injected > 0:
                assert abs(e.recovery_rate - e.n_recovered / e.n_injected) < 1e-3

    def test_zero_injected_nan_rate(self) -> None:
        r = build_completeness_summary([{"tic_id": "1", "n_injected": 0, "n_recovered": 0}])
        entry = r.entries[0]
        assert math.isnan(entry.recovery_rate)

    def test_optional_period_and_depth(self) -> None:
        r = build_completeness_summary([{"tic_id": "1", "n_injected": 10, "n_recovered": 8}])
        assert r.entries[0].mean_period_days is None
        assert r.entries[0].mean_depth_ppm is None

    def test_n_entries_matches_n_targets(self) -> None:
        r = build_completeness_summary(self._targets())
        assert len(r.entries) == r.n_targets

    def test_format_output(self) -> None:
        r = build_completeness_summary(self._targets())
        s = format_completeness_summary(r)
        assert "|" in s
        assert "TIC" in s

    def test_format_empty(self) -> None:
        r = build_completeness_summary([])
        s = format_completeness_summary(r)
        assert "No results" in s
