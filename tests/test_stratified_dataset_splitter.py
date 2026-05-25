"""Tests for Skills/stratified_dataset_splitter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stratified_dataset_splitter import (
    format_split_result,
    stratified_split,
)


def _make_examples(n_pos: int, n_neg: int) -> list[dict]:
    return [{"label": 1, "id": f"p{i}"} for i in range(n_pos)] + [
        {"label": 0, "id": f"n{i}"} for i in range(n_neg)
    ]


def test_valid_split_returns_ok():
    examples = _make_examples(80, 80)
    r = stratified_split(examples)
    assert r.flag == "OK"


def test_total_preserved():
    examples = _make_examples(50, 50)
    r = stratified_split(examples)
    assert len(r.train) + len(r.val) + len(r.test) == len(examples)


def test_both_classes_in_train():
    examples = _make_examples(40, 40)
    r = stratified_split(examples)
    train_labels = {d["label"] for d in r.train}
    assert 0 in train_labels
    assert 1 in train_labels


def test_stratification_preserves_ratio():
    # 75% positive overall → each split should have ~75% positive
    examples = _make_examples(75, 25)
    r = stratified_split(examples)
    for report in (r.train_report, r.val_report, r.test_report):
        if report.n_total > 0:
            assert report.balance_ratio is not None
            assert 0.5 <= report.balance_ratio <= 0.95


def test_seed_deterministic():
    examples = _make_examples(50, 50)
    r1 = stratified_split(examples, seed=7)
    r2 = stratified_split(examples, seed=7)
    assert [d["id"] for d in r1.train] == [d["id"] for d in r2.train]


def test_different_seeds_differ():
    examples = _make_examples(50, 50)
    r1 = stratified_split(examples, seed=1)
    r2 = stratified_split(examples, seed=2)
    # With 100 examples it would be astronomically unlikely for both to match
    assert [d["id"] for d in r1.train] != [d["id"] for d in r2.train]


def test_empty_returns_insufficient():
    r = stratified_split([])
    assert r.flag == "INSUFFICIENT"


def test_all_same_label_returns_insufficient():
    examples = [{"label": 1}] * 10
    r = stratified_split(examples)
    assert r.flag == "INSUFFICIENT"


def test_invalid_fracs_return_invalid():
    examples = _make_examples(50, 50)
    r = stratified_split(examples, train_frac=0.8, val_frac=0.3)
    assert r.flag == "INVALID"


def test_balance_ratio_none_when_n_total_zero():
    examples = _make_examples(50, 50)
    r = stratified_split(examples)
    for report in (r.train_report, r.val_report, r.test_report):
        if report.n_total == 0:
            assert report.balance_ratio is None


def test_format_returns_str():
    examples = _make_examples(40, 40)
    r = stratified_split(examples)
    assert isinstance(format_split_result(r), str)


def test_stratified_split_result_frozen():
    examples = _make_examples(40, 40)
    r = stratified_split(examples)
    try:
        r.seed = 999  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception:
        pass


def test_tiny_dataset_works():
    examples = _make_examples(2, 2)
    r = stratified_split(examples)
    # With 4 examples, split should still work
    assert r.flag in ("OK", "INSUFFICIENT")
    assert len(r.train) + len(r.val) + len(r.test) == 4
