import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from photometric_precision_tracker import PrecisionTrackResult, track_precision, format_precision_track


# --- happy path ---

def test_basic_three_nights():
    result = track_precision([2.0, 3.0, 4.0])
    assert result.n_nights == 3
    assert result.mean_rms_mmag == pytest.approx(3.0, abs=1e-9)
    assert result.flag == "OK"


def test_best_and_worst_indices_correct():
    rms = [3.0, 1.0, 5.0, 2.0]
    result = track_precision(rms)
    assert result.best_night_idx == 1   # index of 1.0
    assert result.worst_night_idx == 2  # index of 5.0


def test_std_computed_correctly():
    rms = [2.0, 4.0]
    result = track_precision(rms)
    # sample std of [2, 4] = sqrt(2) ≈ 1.414
    assert result.std_rms_mmag == pytest.approx(math.sqrt(2.0), abs=1e-9)


def test_single_night_no_std():
    result = track_precision([3.5])
    assert result.n_nights == 1
    assert result.std_rms_mmag == 0.0
    assert result.mean_rms_mmag == pytest.approx(3.5, abs=1e-9)


# --- flag boundary ---

def test_flag_ok_mean_below_threshold():
    result = track_precision([2.0, 3.0, 4.0])
    assert result.flag == "OK"


def test_flag_poor_precision_above_threshold():
    # mean = 8.0 > 5.0 → POOR_PRECISION
    result = track_precision([6.0, 8.0, 10.0])
    assert result.flag == "POOR_PRECISION"


def test_flag_ok_exactly_at_threshold():
    # mean = 5.0 → NOT poor (> 5.0 is the condition)
    result = track_precision([5.0, 5.0, 5.0])
    assert result.flag == "OK"


def test_flag_poor_precision_just_above():
    result = track_precision([5.1, 5.1, 5.1])
    assert result.flag == "POOR_PRECISION"


# --- edge cases ---

def test_empty_returns_no_data():
    result = track_precision([])
    assert result.flag == "NO_DATA"
    assert result.n_nights == 0
    assert result.worst_night_idx == -1
    assert result.best_night_idx == -1


def test_two_nights():
    result = track_precision([1.0, 9.0])
    assert result.n_nights == 2
    assert result.best_night_idx == 0
    assert result.worst_night_idx == 1


# --- return type ---

def test_returns_precision_track_result():
    result = track_precision([2.0, 3.0])
    assert isinstance(result, PrecisionTrackResult)


def test_result_is_frozen():
    result = track_precision([2.0, 3.0])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = track_precision([2.0, 3.0])
    text = format_precision_track(result)
    assert "## Photometric Precision Tracker" in text


def test_format_contains_flag():
    result = track_precision([2.0, 3.0])
    text = format_precision_track(result)
    assert result.flag in text
