"""Tests for Skills/observation_priority_matrix.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_priority_matrix import (
    build_priority_matrix,
    format_priority_matrix,
)


class TestBuildPriorityMatrix:
    def _targets(self) -> list[dict]:
        return [
            {"tic_id": "111", "false_positive_probability": 0.05,
             "detection_confidence": 0.9, "snr": 15.0,
             "n_transits_remaining": 3, "visibility_fraction": 0.8, "tmag": 10.0},
            {"tic_id": "222", "false_positive_probability": 0.50,
             "detection_confidence": 0.4, "snr": 5.0,
             "n_transits_remaining": 1, "visibility_fraction": 0.3, "tmag": 13.5},
        ]

    def test_empty_targets(self) -> None:
        r = build_priority_matrix([])
        assert r.flag == "NO_TARGETS"

    def test_two_targets(self) -> None:
        r = build_priority_matrix(self._targets())
        assert r.n_targets == 2
        assert r.flag == "OK"

    def test_ranks_assigned(self) -> None:
        r = build_priority_matrix(self._targets())
        ranks = [e.rank for e in r.entries]
        assert sorted(ranks) == list(range(1, r.n_targets + 1))

    def test_sorted_by_composite_desc(self) -> None:
        r = build_priority_matrix(self._targets())
        scores = [e.composite_score for e in r.entries]
        assert scores == sorted(scores, reverse=True)

    def test_recommendation_values(self) -> None:
        r = build_priority_matrix(self._targets())
        for e in r.entries:
            assert e.recommendation in ("OBSERVE_NOW", "SCHEDULE", "LOW_PRIORITY")

    def test_science_score_in_range(self) -> None:
        r = build_priority_matrix(self._targets())
        for e in r.entries:
            assert 0.0 <= e.science_score <= 1.0

    def test_feasibility_score_in_range(self) -> None:
        r = build_priority_matrix(self._targets())
        for e in r.entries:
            assert 0.0 <= e.feasibility_score <= 1.0

    def test_urgency_score_in_range(self) -> None:
        r = build_priority_matrix(self._targets())
        for e in r.entries:
            assert 0.0 <= e.urgency_score <= 1.0

    def test_composite_in_range(self) -> None:
        r = build_priority_matrix(self._targets())
        for e in r.entries:
            assert 0.0 <= e.composite_score <= 1.0

    def test_high_quality_target_observe_now(self) -> None:
        target = {"tic_id": "1", "false_positive_probability": 0.01,
                  "detection_confidence": 0.99, "snr": 20.0,
                  "n_transits_remaining": 5, "visibility_fraction": 1.0, "tmag": 8.0}
        r = build_priority_matrix([target])
        assert r.entries[0].recommendation == "OBSERVE_NOW"

    def test_custom_weights(self) -> None:
        r = build_priority_matrix(self._targets(), science_weight=1.0,
                                  feasibility_weight=0.0, urgency_weight=0.0)
        assert r.flag == "OK"

    def test_format_output(self) -> None:
        r = build_priority_matrix(self._targets())
        s = format_priority_matrix(r)
        assert "|" in s
        assert "Rank" in s
