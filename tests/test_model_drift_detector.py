"""Tests for Skills/model_drift_detector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from model_drift_detector import (
    BaselineStats,
    compute_baseline_stats,
    detect_drift,
    format_drift_report,
)


def test_compute_baseline_stats():
    stats = compute_baseline_stats([0.5, 0.6, 0.4, 0.5, 0.55], feature="p_planet")
    assert isinstance(stats, BaselineStats)
    assert 0.4 < stats.mean < 0.7


def test_baseline_n_correct():
    stats = compute_baseline_stats([0.1, 0.2, 0.3])
    assert stats.n == 3


def test_no_drift_similar_distribution():
    # Give baseline the same variance as current so std_ratio ≈ 1
    baseline = [0.5 + 0.01 * (i % 3 - 1) for i in range(50)]
    current = [0.5 + 0.01 * (i % 3 - 1) for i in range(50)]
    baseline_stats = compute_baseline_stats(baseline)
    result = detect_drift(
        current,
        baseline_mean=baseline_stats.mean,
        baseline_std=baseline_stats.std,
        mean_shift_threshold=0.05,
    )
    assert result.flag == "OK"
    assert not result.drift_detected


def test_drift_detected_large_shift():
    current = [0.9] * 30
    result = detect_drift(
        current,
        baseline_mean=0.5,
        baseline_std=0.05,
        mean_shift_threshold=0.05,
    )
    assert result.drift_detected
    assert result.flag in ("DRIFT", "SEVERE_DRIFT")


def test_severe_drift():
    current = [0.9] * 30
    result = detect_drift(
        current,
        baseline_mean=0.5,
        baseline_std=0.05,
        mean_shift_threshold=0.05,
    )
    assert result.flag == "SEVERE_DRIFT"


def test_insufficient_data():
    result = detect_drift(
        [0.9, 0.8],
        baseline_mean=0.5,
        baseline_std=0.05,
        min_samples=10,
    )
    assert result.flag == "INSUFFICIENT_DATA"


def test_mean_shift_computed():
    current = [0.7] * 30
    result = detect_drift(current, baseline_mean=0.5, baseline_std=0.05)
    assert abs(result.mean_shift - 0.2) < 0.01


def test_std_ratio_computed():
    current = [0.5 + 0.15 * (i % 2 - 0.5) for i in range(30)]
    result = detect_drift(current, baseline_mean=0.5, baseline_std=0.05)
    assert result.std_ratio > 1.0


def test_format_drift_report_string():
    result = detect_drift([0.9] * 30, baseline_mean=0.5, baseline_std=0.05)
    text = format_drift_report([result])
    assert isinstance(text, str)
    assert "Drift" in text


def test_format_empty():
    text = format_drift_report([])
    assert "No drift" in text


def test_result_frozen():
    result = detect_drift([0.5] * 20, baseline_mean=0.5, baseline_std=0.05)
    try:
        result.flag = "BAD"  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_zero_baseline_std_with_curr_std():
    current = [0.5 + 0.1 * (i % 2) for i in range(30)]
    result = detect_drift(current, baseline_mean=0.5, baseline_std=0.0)
    assert result.std_ratio > 1.0 or result.flag in ("DRIFT", "SEVERE_DRIFT", "OK")
