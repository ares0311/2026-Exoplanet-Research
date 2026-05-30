import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from detection_threshold_mapper import (
    ThresholdMapResult,
    format_threshold_map,
    map_detection_thresholds,
)

# --- happy path ---

def test_basic_single_period():
    result = map_detection_thresholds([10.0], 100.0, [10], 7.0)
    assert len(result.periods_days) == 1
    assert len(result.min_depths_ppm) == 1
    expected = round(7.0 * 100.0 / math.sqrt(10), 2)
    assert abs(result.min_depths_ppm[0] - expected) < 0.1


def test_multiple_periods_depths_increase_with_fewer_transits():
    periods = [5.0, 10.0, 20.0]
    n_transits = [20, 10, 5]
    result = map_detection_thresholds(periods, 200.0, n_transits, 7.0)
    assert len(result.min_depths_ppm) == 3
    assert result.min_depths_ppm[0] < result.min_depths_ppm[2]


def test_default_snr_threshold():
    result = map_detection_thresholds([5.0], 100.0, [10])
    assert result.snr_threshold == 7.0


def test_custom_snr_threshold():
    result = map_detection_thresholds([5.0], 100.0, [10], snr_threshold=5.0)
    assert result.snr_threshold == 5.0
    expected = round(5.0 * 100.0 / math.sqrt(10), 2)
    assert abs(result.min_depths_ppm[0] - expected) < 0.1


def test_periods_stored_correctly():
    result = map_detection_thresholds([3.0, 7.5], 100.0, [15, 8])
    assert result.periods_days == (3.0, 7.5)
    assert result.n_transits == (15, 8)


# --- flag boundary ---

def test_flag_ok_low_depths():
    result = map_detection_thresholds([5.0], 100.0, [100], 7.0)
    assert result.flag == "OK"


def test_flag_shallow_coverage():
    # noise=5000, n=1, snr=7 → depth=35000 > 5000
    result = map_detection_thresholds([100.0], 5000.0, [1], 7.0)
    assert result.flag == "SHALLOW_COVERAGE"


def test_flag_ok_all_below_5000():
    result = map_detection_thresholds([1.0, 2.0], 100.0, [100, 200], 7.0)
    assert result.flag == "OK"


# --- edge cases ---

def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        map_detection_thresholds([1.0, 2.0], 100.0, [5], 7.0)


def test_zero_transits_clamped_to_one():
    result = map_detection_thresholds([10.0], 100.0, [0], 7.0)
    expected = round(7.0 * 100.0, 2)
    assert abs(result.min_depths_ppm[0] - expected) < 0.1


def test_single_transit():
    result = map_detection_thresholds([30.0], 100.0, [1], 7.0)
    assert abs(result.min_depths_ppm[0] - 700.0) < 0.1


# --- return type and immutability ---

def test_returns_threshold_map_result():
    result = map_detection_thresholds([5.0], 100.0, [10])
    assert isinstance(result, ThresholdMapResult)


def test_result_is_frozen():
    result = map_detection_thresholds([5.0], 100.0, [10])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


def test_tuple_field_types():
    result = map_detection_thresholds([5.0, 10.0], 100.0, [10, 5])
    assert isinstance(result.periods_days, tuple)
    assert isinstance(result.min_depths_ppm, tuple)
    assert isinstance(result.n_transits, tuple)


# --- format output ---

def test_format_contains_header():
    result = map_detection_thresholds([5.0], 100.0, [10])
    text = format_threshold_map(result)
    assert "## Detection Threshold Map" in text


def test_format_contains_flag():
    result = map_detection_thresholds([5.0], 100.0, [10])
    text = format_threshold_map(result)
    assert result.flag in text


def test_format_contains_period_in_table():
    result = map_detection_thresholds([5.0], 100.0, [10])
    text = format_threshold_map(result)
    assert "5.000" in text
