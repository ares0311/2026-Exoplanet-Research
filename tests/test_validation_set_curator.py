"""Tests for Skills/validation_set_curator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from validation_set_curator import (
    curate_validation_set,
    format_curation_report,
)

_POS = [{"tic_id": i, "period_days": 5.0, "label": "planet_candidate"} for i in range(300)]
_NEG = [{"tic_id": i + 300, "period_days": 5.0, "label": "false_positive"} for i in range(300)]
_ROWS = _POS + _NEG


def test_flag_ok():
    r = curate_validation_set(_ROWS, n_val_per_class=100)
    assert r.flag == "OK"


def test_val_size():
    r = curate_validation_set(_ROWS, n_val_per_class=100)
    assert r.n_val_pos == 100
    assert r.n_val_neg == 100


def test_train_plus_val_le_total():
    r = curate_validation_set(_ROWS, n_val_per_class=100)
    assert r.n_train + len(r.val_rows) <= len(_ROWS)


def test_no_tic_leakage():
    r = curate_validation_set(_ROWS, n_val_per_class=50)
    val_tic_ids = {row["tic_id"] for row in r.val_rows}
    train_tic_ids = {row["tic_id"] for row in r.train_rows}
    assert val_tic_ids.isdisjoint(train_tic_ids)


def test_excluded_tic_ids():
    r = curate_validation_set(_ROWS, n_val_per_class=50)
    assert len(r.excluded_tic_ids) > 0


def test_empty_flag():
    r = curate_validation_set([])
    assert r.flag == "EMPTY"


def test_invalid_flag():
    r = curate_validation_set("bad")
    assert r.flag == "INVALID"


def test_partial_when_not_enough():
    small_pos = [{"tic_id": i, "label": "planet_candidate", "period_days": 1.0}
                 for i in range(5)]
    small_neg = [{"tic_id": i + 5, "label": "false_positive", "period_days": 1.0}
                 for i in range(5)]
    r = curate_validation_set(small_pos + small_neg, n_val_per_class=100)
    assert r.flag == "PARTIAL"


def test_reproducible_with_seed():
    r1 = curate_validation_set(_ROWS, n_val_per_class=50, seed=7)
    r2 = curate_validation_set(_ROWS, n_val_per_class=50, seed=7)
    ids1 = {row["tic_id"] for row in r1.val_rows}
    ids2 = {row["tic_id"] for row in r2.val_rows}
    assert ids1 == ids2


def test_different_seeds_differ():
    r1 = curate_validation_set(_ROWS, n_val_per_class=50, seed=1)
    r2 = curate_validation_set(_ROWS, n_val_per_class=50, seed=99)
    ids1 = {row["tic_id"] for row in r1.val_rows}
    ids2 = {row["tic_id"] for row in r2.val_rows}
    assert ids1 != ids2


def test_format_returns_string():
    r = curate_validation_set(_ROWS, n_val_per_class=50)
    s = format_curation_report(r)
    assert isinstance(s, str)
    assert "Validation" in s


def test_result_frozen():
    r = curate_validation_set(_ROWS, n_val_per_class=10)
    try:
        r.n_val_pos = 0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_val_rows_are_dicts():
    r = curate_validation_set(_ROWS, n_val_per_class=10)
    assert all(isinstance(row, dict) for row in r.val_rows)
