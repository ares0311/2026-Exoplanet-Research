"""Tests for Skills/training_data_stats_reporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from training_data_stats_reporter import (
    TrainingDataStats,
    compute_training_stats,
    format_training_stats,
)

_ROWS = [
    {"label": "planet_candidate", "period_days": 5.0, "depth_ppm": 10000, "source": "toi"},
    {"label": "planet_candidate", "period_days": 10.0, "depth_ppm": 5000, "source": "toi"},
    {"label": "false_positive", "period_days": 3.0, "depth_ppm": 2000, "source": "koi"},
    {"label": "false_positive", "period_days": 7.0, "depth_ppm": 8000, "source": "koi"},
]


def test_flag_ok():
    s = compute_training_stats(_ROWS)
    assert s.flag == "OK"


def test_n_positive():
    s = compute_training_stats(_ROWS)
    assert s.n_positive == 2


def test_n_negative():
    s = compute_training_stats(_ROWS)
    assert s.n_negative == 2


def test_n_total():
    s = compute_training_stats(_ROWS)
    assert s.n_total == 4


def test_period_min():
    s = compute_training_stats(_ROWS)
    assert abs(s.period_min - 3.0) < 1e-9


def test_period_max():
    s = compute_training_stats(_ROWS)
    assert abs(s.period_max - 10.0) < 1e-9


def test_period_median():
    s = compute_training_stats(_ROWS)
    assert s.period_median is not None
    assert 3.0 <= s.period_median <= 10.0


def test_depth_min():
    s = compute_training_stats(_ROWS)
    assert abs(s.depth_ppm_min - 2000) < 1e-9


def test_source_counts():
    s = compute_training_stats(_ROWS)
    assert "toi" in s.source_counts
    assert s.source_counts["toi"] == 2


def test_empty_flag():
    s = compute_training_stats([])
    assert s.flag == "EMPTY"


def test_invalid_flag():
    s = compute_training_stats("bad")
    assert s.flag == "INVALID"


def test_format_returns_string():
    s = compute_training_stats(_ROWS)
    text = format_training_stats(s)
    assert isinstance(text, str)
    assert "Statistics" in text


def test_result_frozen():
    s = compute_training_stats(_ROWS)
    try:
        s.n_total = 0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
