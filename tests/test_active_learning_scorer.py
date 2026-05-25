"""Tests for Skills/active_learning_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from active_learning_scorer import (
    format_active_learning,
    rank_by_uncertainty,
)


def test_empty_returns_empty():
    r = rank_by_uncertainty([])
    assert r.flag == "EMPTY"


def test_single_candidate_n_returned_is_1():
    r = rank_by_uncertainty([{"tic_id": "TIC1", "score": 0.5}])
    assert r.n_returned == 1


def test_most_uncertain_is_rank_1():
    candidates = [
        {"tic_id": "TIC1", "score": 0.9},   # very certain
        {"tic_id": "TIC2", "score": 0.51},  # most uncertain
        {"tic_id": "TIC3", "score": 0.1},   # very certain
    ]
    r = rank_by_uncertainty(candidates)
    rank1 = next(e for e in r.entries if e.rank == 1)
    assert rank1.tic_id == "TIC2"


def test_most_certain_is_last():
    candidates = [
        {"tic_id": "TIC1", "score": 0.5},   # most uncertain
        {"tic_id": "TIC2", "score": 0.99},  # most certain
    ]
    r = rank_by_uncertainty(candidates)
    last = r.entries[-1]
    assert last.tic_id == "TIC2"


def test_entropy_zero_at_score_0():
    r = rank_by_uncertainty([{"tic_id": "T1", "score": 0.0}])
    assert abs(r.entries[0].entropy) < 1e-9


def test_entropy_zero_at_score_1():
    r = rank_by_uncertainty([{"tic_id": "T1", "score": 1.0}])
    assert abs(r.entries[0].entropy) < 1e-9


def test_entropy_max_at_score_half():
    r = rank_by_uncertainty([{"tic_id": "T1", "score": 0.5}])
    assert abs(r.entries[0].entropy - 1.0) < 1e-9


def test_top_n_returns_n():
    candidates = [{"tic_id": f"TIC{i}", "score": 0.5 + i * 0.01} for i in range(10)]
    r = rank_by_uncertainty(candidates, top_n=3)
    assert r.n_returned == 3


def test_top_n_greater_than_n_returns_all():
    candidates = [{"tic_id": "T1", "score": 0.5}, {"tic_id": "T2", "score": 0.8}]
    r = rank_by_uncertainty(candidates, top_n=100)
    assert r.n_returned == 2


def test_uncertainty_equals_abs_score_minus_half():
    score = 0.73
    r = rank_by_uncertainty([{"tic_id": "T", "score": score}])
    assert abs(r.entries[0].uncertainty - abs(score - 0.5)) < 1e-9


def test_ranks_are_1_based_contiguous():
    candidates = [{"tic_id": f"T{i}", "score": float(i) / 10} for i in range(5)]
    r = rank_by_uncertainty(candidates)
    ranks = [e.rank for e in r.entries]
    assert sorted(ranks) == list(range(1, len(candidates) + 1))


def test_active_learning_result_frozen():
    r = rank_by_uncertainty([{"tic_id": "T", "score": 0.5}])
    try:
        r.n_candidates = 999  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception:
        pass


def test_format_returns_str():
    r = rank_by_uncertainty([{"tic_id": "T", "score": 0.5}])
    assert isinstance(format_active_learning(r), str)


def test_non_dict_in_list_returns_invalid():
    r = rank_by_uncertainty(["not_a_dict"])  # type: ignore[list-item]
    assert r.flag == "INVALID"
