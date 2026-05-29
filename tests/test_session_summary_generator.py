"""Tests for Skills/session_summary_generator.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from session_summary_generator import (
    SessionSummary,
    build_session_summary,
    format_session_summary,
)


def _candidate(tic_id=100, fpp=0.05, pathway="tfop_ready"):
    return {
        "tic_id": tic_id,
        "period_days": 3.0,
        "status": "candidate_found",
        "false_positive_probability": fpp,
        "pathway": pathway,
    }


def _error_row(tic_id=200):
    return {"tic_id": tic_id, "status": "error"}


def _no_data_row(tic_id=300):
    return {"tic_id": tic_id, "status": "no_data"}


def test_empty_input():
    result = build_session_summary([])
    assert result.flag == "EMPTY"
    assert result.n_scanned == 0


def test_single_candidate():
    result = build_session_summary([_candidate()])
    assert result.n_candidates == 1
    assert result.n_scanned == 1


def test_counts_errors():
    rows = [_candidate(), _error_row(), _error_row()]
    result = build_session_summary(rows)
    assert result.n_errors == 2


def test_counts_no_data():
    rows = [_no_data_row(), _no_data_row(), _candidate()]
    result = build_session_summary(rows)
    assert result.n_no_data == 2


def test_high_error_rate_flag():
    rows = [_candidate()] + [_error_row(i) for i in range(200, 210)]
    result = build_session_summary(rows)
    assert result.flag == "HIGH_ERROR_RATE"


def test_top_candidates_sorted_by_fpp():
    rows = [_candidate(100, fpp=0.30), _candidate(200, fpp=0.05), _candidate(300, fpp=0.20)]
    result = build_session_summary(rows, top_n=3)
    fpps = [r.get("false_positive_probability") for r in result.top_candidates]
    assert fpps == sorted(fpps)


def test_top_n_limit():
    rows = [_candidate(i, fpp=0.1) for i in range(100, 110)]
    result = build_session_summary(rows, top_n=3)
    assert len(result.top_candidates) == 3


def test_session_id_stored():
    result = build_session_summary([_candidate()], session_id="run-42")
    assert result.session_id == "run-42"


def test_elapsed_stored():
    result = build_session_summary([_candidate()], elapsed_s=123.4)
    assert result.elapsed_s == 123.4


def test_returns_session_summary_type():
    result = build_session_summary([_candidate()])
    assert isinstance(result, SessionSummary)


def test_format_contains_flag():
    result = build_session_summary([_candidate()])
    md = format_session_summary(result)
    assert result.flag in md


def test_format_contains_next_steps():
    result = build_session_summary([_candidate()])
    md = format_session_summary(result)
    assert "Next Steps" in md


def test_scores_dict_fpp():
    row = {
        "tic_id": 1,
        "status": "candidate_found",
        "scores": {"false_positive_probability": 0.08},
    }
    result = build_session_summary([row])
    assert result.n_candidates == 1
