"""Tests for Skills/candidate_significance_ranker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_significance_ranker import (
    format_significance_table,
    rank_by_significance,
)

_ROWS = [
    {"tic_id": 100, "period_days": 5.0, "snr": 20.0,
     "false_positive_probability": 0.05, "novelty_score": 0.9},
    {"tic_id": 200, "period_days": 10.0, "snr": 8.0,
     "false_positive_probability": 0.40, "novelty_score": 0.5},
    {"tic_id": 300, "period_days": 3.0, "snr": 5.0,
     "false_positive_probability": 0.80, "novelty_score": 0.2},
]


def test_returns_list():
    results = rank_by_significance(_ROWS)
    assert isinstance(results, list)


def test_sorted_descending():
    results = rank_by_significance(_ROWS)
    scores = [r.significance_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_n():
    results = rank_by_significance(_ROWS, top_n=2)
    assert len(results) <= 2


def test_min_snr_filter():
    results = rank_by_significance(_ROWS, min_snr=10.0)
    for r in results:
        if r.flag == "FILTERED":
            assert r.snr is None or r.snr < 10.0


def test_max_fpp_filter():
    results = rank_by_significance(_ROWS, max_fpp=0.50)
    non_filtered = [r for r in results if r.flag != "FILTERED"]
    for r in non_filtered:
        assert r.fpp is None or r.fpp <= 0.50


def test_rank_assigned():
    results = rank_by_significance(_ROWS)
    non_filtered = [r for r in results if r.rank > 0]
    assert non_filtered[0].rank == 1


def test_high_fpp_flag():
    row = {"tic_id": 999, "snr": 15.0,
           "false_positive_probability": 0.80, "novelty_score": 0.5}
    results = rank_by_significance([row])
    assert results[0].flag == "HIGH_FPP"


def test_low_snr_flag():
    row = {"tic_id": 999, "snr": 5.0,
           "false_positive_probability": 0.10, "novelty_score": 0.5}
    results = rank_by_significance([row])
    assert results[0].flag == "LOW_SNR"


def test_ok_flag_good_candidate():
    row = {"tic_id": 1, "snr": 20.0,
           "false_positive_probability": 0.05, "novelty_score": 0.8}
    results = rank_by_significance([row])
    assert results[0].flag == "OK"


def test_scores_from_nested_scores_dict():
    row = {
        "tic_id": 1,
        "snr": 15.0,
        "scores": {"false_positive_probability": 0.10, "novelty_score": 0.7},
    }
    results = rank_by_significance([row])
    assert results[0].fpp == 0.10


def test_empty_input():
    results = rank_by_significance([])
    assert results == []


def test_format_returns_string():
    results = rank_by_significance(_ROWS)
    text = format_significance_table(results)
    assert isinstance(text, str)


def test_format_contains_header():
    results = rank_by_significance(_ROWS)
    text = format_significance_table(results)
    assert "Significance" in text


def test_format_empty():
    text = format_significance_table([])
    assert "No results" in text
