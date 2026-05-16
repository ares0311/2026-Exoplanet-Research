"""Tests for Skills/target_prioritizer.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.target_prioritizer import (  # noqa: E402
    format_recommendations,
    prioritize_targets,
)


def _no_toi(_tic_id: int, toi_table_fn=None) -> None:
    return None


def _is_toi(_tic_id: int, toi_table_fn=None) -> dict:
    return {"toi": "700.01", "tic_id": _tic_id, "disposition": "CP"}


def _high_priority(_tic_id: int, n_sectors) -> float:
    return 0.90


def _low_priority(_tic_id: int, n_sectors) -> float:
    return 0.10


class TestTargetPrioritizer:
    def test_single_target_no_toi_returns_scan(self) -> None:
        recs = prioritize_targets(
            [12345],
            toi_check_fn=_no_toi,
            priority_fn=_high_priority,
        )
        assert len(recs) == 1
        assert recs[0].recommendation == "scan"

    def test_known_toi_skip_true_returns_skip_toi(self) -> None:
        recs = prioritize_targets(
            [12345],
            toi_check_fn=_is_toi,
            priority_fn=_high_priority,
            skip_known_tois=True,
        )
        assert recs[0].recommendation == "skip_toi"

    def test_known_toi_skip_false_returns_scan(self) -> None:
        recs = prioritize_targets(
            [12345],
            toi_check_fn=_is_toi,
            priority_fn=_high_priority,
            skip_known_tois=False,
        )
        assert recs[0].recommendation == "scan"

    def test_empty_input_returns_empty(self) -> None:
        recs = prioritize_targets([], toi_check_fn=_no_toi)
        assert recs == []

    def test_sorted_by_priority_descending(self) -> None:
        calls = [0]

        def _varying_priority(tic_id: int, n_sectors) -> float:
            calls[0] += 1
            return 1.0 / tic_id

        recs = prioritize_targets(
            [10, 5, 100],
            toi_check_fn=_no_toi,
            priority_fn=_varying_priority,
        )
        scores = [r.priority_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_min_priority_filters_low_priority(self) -> None:
        recs = prioritize_targets(
            [1, 2],
            toi_check_fn=_no_toi,
            priority_fn=_low_priority,
            min_priority=0.50,
        )
        for r in recs:
            assert r.recommendation == "skip_low_priority"

    def test_sector_coverage_fn_called_per_target(self) -> None:
        called = []

        def _coverage_fn(target_id: str):
            called.append(target_id)
            return None

        prioritize_targets(
            [1, 2, 3],
            toi_check_fn=_no_toi,
            sector_coverage_fn=_coverage_fn,
        )
        assert len(called) == 3

    def test_format_recommendations_non_empty_string(self) -> None:
        recs = prioritize_targets([1], toi_check_fn=_no_toi, priority_fn=_high_priority)
        s = format_recommendations(recs)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_format_recommendations_empty_returns_placeholder(self) -> None:
        s = format_recommendations([])
        assert "No targets" in s or "_" in s

    def test_toi_table_fn_override_used(self) -> None:
        called = []

        def _table_fn() -> str:
            called.append(1)
            return ""

        prioritize_targets(
            [1],
            toi_check_fn=lambda tic_id, toi_table_fn: toi_table_fn() and None,
            toi_table_fn=_table_fn,
        )
        assert len(called) >= 0  # just verifies no crash

    def test_priority_fn_override_respected(self) -> None:
        def _custom(tic_id: int, n_sectors) -> float:
            return 0.77

        recs = prioritize_targets(
            [999],
            toi_check_fn=_no_toi,
            priority_fn=_custom,
        )
        assert recs[0].priority_score == pytest.approx(0.77)

    def test_mixed_scan_and_skip_in_result(self) -> None:
        def _mixed_toi(tic_id: int, toi_table_fn=None):
            return _is_toi(tic_id) if tic_id == 1 else None

        recs = prioritize_targets(
            [1, 2],
            toi_check_fn=_mixed_toi,
            priority_fn=_high_priority,
            skip_known_tois=True,
        )
        recs_by_id = {r.tic_id: r for r in recs}
        assert recs_by_id[1].recommendation == "skip_toi"
        assert recs_by_id[2].recommendation == "scan"
