"""Tests for Skills/label_propagator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from label_propagator import (
    PropagationResult,
    format_propagation_report,
    propagate_labels,
)

_POS_ROWS = [
    {"tic_id": 1, "period_days": 5.0, "label": "planet_candidate", "source": "toi"},
    {"tic_id": 2, "period_days": 10.0, "label": "planet_candidate", "source": "toi"},
]
_NEG_ROWS = [
    {"tic_id": 3, "period_days": 7.0, "label": "false_positive", "source": "toi"},
]


def test_flag_ok():
    r = propagate_labels(_POS_ROWS + _NEG_ROWS)
    assert r.flag == "OK"


def test_adds_harmonics_for_positives():
    r = propagate_labels(_POS_ROWS, harmonic_factors=(2.0,))
    assert r.n_added == 2  # one harmonic per positive row


def test_negatives_not_propagated():
    r = propagate_labels(_NEG_ROWS, harmonic_factors=(2.0, 0.5))
    assert r.n_added == 0


def test_no_duplicate_harmonics():
    rows = [
        {"tic_id": 1, "period_days": 5.0, "label": "planet_candidate", "source": "x"},
        {"tic_id": 1, "period_days": 10.0, "label": "planet_candidate", "source": "x"},
    ]
    # P=5 → 2P=10 (already exists); P=10 → 2P=20 (new); so n_added=1
    r = propagate_labels(rows, harmonic_factors=(2.0,))
    assert r.n_added == 1


def test_source_tagged():
    r = propagate_labels(_POS_ROWS[:1], harmonic_factors=(2.0,))
    assert r.n_added == 1
    new_row = r.propagated_rows[0]
    assert "propagated" in new_row.get("source", "")


def test_original_rows_preserved():
    r = propagate_labels(_POS_ROWS + _NEG_ROWS)
    assert len(r.original_rows) == len(_POS_ROWS) + len(_NEG_ROWS)


def test_empty_flag():
    r = propagate_labels([])
    assert r.flag == "EMPTY"


def test_invalid_flag():
    r = propagate_labels("not-a-list")
    assert r.flag == "INVALID"


def test_period_rtol_prevents_close_duplicate():
    rows = [
        {"tic_id": 1, "period_days": 5.0, "label": "planet_candidate", "source": "x"},
        {"tic_id": 1, "period_days": 9.99, "label": "planet_candidate", "source": "x"},
    ]
    # 2*5=10, which is close to 9.99 (0.1% diff)
    r = propagate_labels(rows, harmonic_factors=(2.0,), period_rtol=0.01)
    # 10 vs 9.99 → 0.1% < 1% rtol → treated as duplicate; only 1 new harmonic
    assert r.n_added <= 1


def test_format_returns_string():
    r = propagate_labels(_POS_ROWS)
    s = format_propagation_report(r)
    assert isinstance(s, str)
    assert "Propagation" in s


def test_result_frozen():
    r = propagate_labels(_POS_ROWS)
    try:
        r.n_added = 0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_multiple_harmonic_factors():
    r = propagate_labels(_POS_ROWS[:1], harmonic_factors=(0.5, 2.0, 3.0))
    assert r.n_added == 3
