"""Tests for Skills/target_selection_optimizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from target_selection_optimizer import (
    TargetScore,
    format_selection_result,
    optimize_target_selection,
)

_TARGETS = [
    {"tic_id": 1, "false_positive_probability": 0.05, "novelty_score": 0.9,
     "snr": 20.0, "pathway": "tfop_ready",
     "scores": {"detection_confidence": 0.95}, "provenance_score": 0.9},
    {"tic_id": 2, "false_positive_probability": 0.50, "novelty_score": 0.4,
     "snr": 7.0, "pathway": "github_only_reproducibility",
     "scores": {"detection_confidence": 0.50}, "provenance_score": 0.5},
    {"tic_id": 3, "false_positive_probability": 0.80, "novelty_score": 0.2,
     "snr": 5.0, "pathway": "github_only_reproducibility",
     "scores": {"detection_confidence": 0.30}, "provenance_score": 0.4},
]


def test_returns_list():
    results = optimize_target_selection(_TARGETS)
    assert isinstance(results, list)


def test_sorted_descending():
    results = optimize_target_selection(_TARGETS)
    scores = [r.composite_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_n():
    results = optimize_target_selection(_TARGETS, top_n=2)
    assert len(results) <= 2


def test_rank_starts_at_one():
    results = optimize_target_selection(_TARGETS)
    assert results[0].rank == 1


def test_recommended_flag_high_score():
    results = optimize_target_selection(_TARGETS)
    best = results[0]
    # Best target should be recommended given high FPP=0.05, novelty=0.9
    assert best.flag in ("RECOMMENDED", "MARGINAL")


def test_skip_flag_low_score():
    results = optimize_target_selection(_TARGETS)
    worst = results[-1]
    assert worst.flag in ("SKIP", "MARGINAL")


def test_min_composite_score_filter():
    results = optimize_target_selection(_TARGETS, min_composite_score=0.99)
    assert len(results) == 0


def test_composite_score_bounded():
    results = optimize_target_selection(_TARGETS)
    for r in results:
        assert 0.0 <= r.composite_score <= 1.0


def test_tic_id_stored():
    results = optimize_target_selection(_TARGETS)
    tics = {r.tic_id for r in results}
    assert tics == {1, 2, 3}


def test_empty_input():
    results = optimize_target_selection([])
    assert results == []


def test_format_returns_string():
    results = optimize_target_selection(_TARGETS)
    text = format_selection_result(results)
    assert isinstance(text, str)
    assert "Optimizer" in text


def test_format_empty():
    text = format_selection_result([])
    assert "No targets" in text


def test_custom_weights():
    results = optimize_target_selection(_TARGETS, science_weight=1.0,
                                        obs_weight=0.0, stellar_weight=0.0,
                                        pipeline_weight=0.0)
    assert results[0].rank == 1
