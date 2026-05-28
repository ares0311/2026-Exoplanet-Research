"""Tests for Skills/snippet_deduplicator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from snippet_deduplicator import (
    DeduplicationResult,
    apply_deduplication,
    deduplicate_snippets,
    format_dedup_report,
)

_ROWS = [
    {"tic_id": 1, "period_days": 5.0, "label": "planet_candidate"},
    {"tic_id": 1, "period_days": 5.01, "label": "planet_candidate"},  # dup
    {"tic_id": 2, "period_days": 10.0, "label": "false_positive"},
    {"tic_id": 3, "period_days": 7.0, "label": "planet_candidate"},
]


def test_removes_duplicate():
    r = deduplicate_snippets(_ROWS)
    assert r.n_removed == 1


def test_output_less_than_input():
    r = deduplicate_snippets(_ROWS)
    assert r.n_output < r.n_input


def test_flag_ok():
    r = deduplicate_snippets(_ROWS)
    assert r.flag == "OK"


def test_empty_flag():
    r = deduplicate_snippets([])
    assert r.flag == "EMPTY"


def test_invalid_flag():
    r = deduplicate_snippets("not-a-list")
    assert r.flag == "INVALID"


def test_no_duplicates():
    rows = [
        {"tic_id": 1, "period_days": 5.0},
        {"tic_id": 2, "period_days": 5.0},
        {"tic_id": 3, "period_days": 5.0},
    ]
    r = deduplicate_snippets(rows)
    assert r.n_removed == 0


def test_dup_tic_ids_recorded():
    r = deduplicate_snippets(_ROWS)
    assert 1 in r.duplicate_tic_ids


def test_apply_returns_list():
    rows = apply_deduplication(_ROWS)
    assert isinstance(rows, list)
    assert len(rows) == 3


def test_apply_does_not_mutate():
    original = list(_ROWS)
    apply_deduplication(_ROWS)
    assert len(_ROWS) == len(original)


def test_strict_tolerance():
    rows = [
        {"tic_id": 1, "period_days": 5.0},
        {"tic_id": 1, "period_days": 5.1},  # 2% difference
    ]
    r = deduplicate_snippets(rows, period_rtol=0.01)
    assert r.n_removed == 0   # 2% > 1% tolerance → not a dup


def test_loose_tolerance():
    rows = [
        {"tic_id": 1, "period_days": 5.0},
        {"tic_id": 1, "period_days": 5.05},
    ]
    r = deduplicate_snippets(rows, period_rtol=0.05)
    assert r.n_removed == 1


def test_format_returns_string():
    r = deduplicate_snippets(_ROWS)
    s = format_dedup_report(r)
    assert isinstance(s, str)
    assert "Deduplication" in s


def test_result_frozen():
    r = deduplicate_snippets(_ROWS)
    try:
        r.n_removed = 0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
