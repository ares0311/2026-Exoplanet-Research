"""Tests for Skills/training_data_monitor.py (13 tests)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from training_data_monitor import (
    format_monitor_result,
    monitor_from_path,
    monitor_training_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snippet(label, source="toi", snr=None, depth_ppm=None):
    d = {"label": label, "source": source}
    if snr is not None:
        d["snr"] = snr
    if depth_ppm is not None:
        d["depth_ppm"] = depth_ppm
    return d


def _make_snippets(n_pos, n_neg, snr=10.0, depth=1000.0):
    snippets = [_snippet(1, snr=snr, depth_ppm=depth) for _ in range(n_pos)]
    snippets += [_snippet(0, snr=snr, depth_ppm=depth) for _ in range(n_neg)]
    return snippets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gate_open():
    snippets = _make_snippets(3000, 3000)
    result = monitor_training_data(snippets, label_threshold=5000, max_balance_ratio=3.0)
    assert result.gate_open is True
    assert result.flag == "OK"


def test_gate_closed_insufficient_labels():
    snippets = _make_snippets(100, 100)
    result = monitor_training_data(snippets, label_threshold=5000)
    assert result.gate_open is False
    assert result.flag == "INSUFFICIENT"


def test_gate_closed_bad_balance():
    # 1 positive, 4 negatives → ratio 4.0 > max 3.0
    snippets = _make_snippets(1000, 4000)
    result = monitor_training_data(snippets, label_threshold=4000, max_balance_ratio=3.0)
    assert result.gate_open is False


def test_n_positive_n_negative():
    snippets = _make_snippets(100, 200)
    result = monitor_training_data(snippets, label_threshold=10)
    assert result.n_positive == 100
    assert result.n_negative == 200


def test_balance_ratio():
    snippets = _make_snippets(100, 200)
    result = monitor_training_data(snippets, label_threshold=10)
    assert result.balance_ratio == pytest.approx(2.0)


def test_no_positives_closes_gate():
    snippets = [_snippet(0)] * 100
    result = monitor_training_data(snippets, label_threshold=50)
    assert result.gate_open is False
    assert result.balance_ratio is None


def test_source_breakdown():
    snippets = [_snippet(1, source="toi")] * 3 + [_snippet(0, source="koi")] * 2
    result = monitor_training_data(snippets, label_threshold=1)
    assert result.source_breakdown.get("toi", 0) == 3
    assert result.source_breakdown.get("koi", 0) == 2


def test_snr_percentiles():
    snippets = [_snippet(i % 2, snr=float(i)) for i in range(1, 11)]
    result = monitor_training_data(snippets, label_threshold=1)
    assert result.snr_p50 is not None


def test_depth_percentiles():
    snippets = [_snippet(i % 2, depth_ppm=float(i * 100)) for i in range(1, 11)]
    result = monitor_training_data(snippets, label_threshold=1)
    assert result.depth_p50 is not None


def test_invalid_input():
    result = monitor_training_data("not a list")
    assert result.flag == "INVALID"
    assert result.gate_open is False


def test_from_path_ok(tmp_path):
    snippets = _make_snippets(3000, 3000)
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(snippets))
    result = monitor_from_path(p, label_threshold=5000)
    assert result.gate_open is True


def test_from_path_missing_file(tmp_path):
    result = monitor_from_path(tmp_path / "nonexistent.json")
    assert result.flag == "INVALID"


def test_formatter_contains_gate():
    snippets = _make_snippets(3000, 3000)
    result = monitor_training_data(snippets, label_threshold=5000)
    text = format_monitor_result(result)
    assert "YES" in text or "NO" in text
    assert len(text) > 0
