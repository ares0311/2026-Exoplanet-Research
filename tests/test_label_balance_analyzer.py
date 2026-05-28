"""Tests for Skills/label_balance_analyzer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from label_balance_analyzer import (
    LabelBalanceResult,
    analyze_label_balance,
    format_balance_report,
)

_POS = [{"label": "planet_candidate"}] * 100
_NEG = [{"label": "false_positive"}] * 100


def test_balanced_ok():
    r = analyze_label_balance(_POS + _NEG)
    assert r.flag == "OK"


def test_n_positive():
    r = analyze_label_balance(_POS + _NEG)
    assert r.n_positive == 100


def test_n_negative():
    r = analyze_label_balance(_POS + _NEG)
    assert r.n_negative == 100


def test_n_total():
    r = analyze_label_balance(_POS + _NEG)
    assert r.n_total == 200


def test_ratio_one():
    r = analyze_label_balance(_POS + _NEG)
    assert abs(r.ratio - 1.0) < 1e-9


def test_class_weights_balanced():
    r = analyze_label_balance(_POS + _NEG)
    # With 100 pos and 100 neg: w = 200 / (2*100) = 1.0
    assert abs(r.class_weight_pos - 1.0) < 1e-9
    assert abs(r.class_weight_neg - 1.0) < 1e-9


def test_imbalanced_flag():
    heavy_neg = [{"label": "false_positive"}] * 1000
    r = analyze_label_balance(_POS + heavy_neg, imbalance_threshold=5.0)
    assert r.flag == "IMBALANCED"


def test_empty_flag():
    r = analyze_label_balance([])
    assert r.flag == "EMPTY"


def test_invalid_flag():
    r = analyze_label_balance("not-a-list")
    assert r.flag == "INVALID"


def test_class_weight_pos_higher_when_minority():
    few_pos = [{"label": "planet_candidate"}] * 10
    r = analyze_label_balance(few_pos + _NEG)
    assert r.class_weight_pos > r.class_weight_neg


def test_format_returns_string():
    r = analyze_label_balance(_POS + _NEG)
    s = format_balance_report(r)
    assert isinstance(s, str)
    assert "Balance" in s


def test_format_warning_on_imbalanced():
    heavy_neg = [{"label": "false_positive"}] * 1000
    r = analyze_label_balance(_POS + heavy_neg)
    s = format_balance_report(r)
    assert "Warning" in s


def test_result_frozen():
    r = analyze_label_balance(_POS + _NEG)
    try:
        r.n_positive = 99  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
