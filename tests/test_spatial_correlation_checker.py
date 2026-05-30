import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from spatial_correlation_checker import (
    SpatialCorrResult,
    check_spatial_correlation,
    format_spatial_corr,
)

# --- happy path ---

def test_perfect_positive_correlation():
    # x == y → Pearson r = 1.0
    vals = [float(i) for i in range(20)]
    result = check_spatial_correlation(vals, vals)
    assert result.pearson_r == pytest.approx(1.0, abs=1e-9)
    assert result.correlated is True
    assert result.flag == "SPATIAL_CORRELATION"


def test_perfect_negative_correlation():
    x = [float(i) for i in range(20)]
    y = [float(19 - i) for i in range(20)]
    result = check_spatial_correlation(x, y)
    assert result.pearson_r == pytest.approx(-1.0, abs=1e-9)
    assert result.abs_r == pytest.approx(1.0, abs=1e-9)
    assert result.correlated is True
    assert result.flag == "SPATIAL_CORRELATION"


def test_no_correlation_orthogonal():
    # x constant → denominator zero → r = 0
    x = [1.0] * 20
    y = [float(i) for i in range(20)]
    result = check_spatial_correlation(x, y)
    assert result.pearson_r == pytest.approx(0.0, abs=1e-9)
    assert result.correlated is False
    assert result.flag == "OK"


def test_n_points_is_min_of_inputs():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0]
    result = check_spatial_correlation(x, y)
    assert result.n_points == 2


# --- flag boundary ---

def test_flag_ok_low_abs_r():
    # Uncorrelated-ish: alternating sign
    x = [1.0, -1.0] * 10
    y = [1.0] * 20  # constant y → r = 0
    result = check_spatial_correlation(x, y)
    assert result.correlated is False
    assert result.flag == "OK"


def test_flag_spatial_correlation_above_0_5():
    # Strong monotonic → |r| >> 0.5
    x = [float(i) for i in range(30)]
    y = [float(i) for i in range(30)]
    result = check_spatial_correlation(x, y)
    assert result.abs_r > 0.5
    assert result.flag == "SPATIAL_CORRELATION"


def test_abs_r_equals_abs_pearson_r():
    x = [float(i) for i in range(10)]
    y = [float(9 - i) for i in range(10)]
    result = check_spatial_correlation(x, y)
    assert result.abs_r == pytest.approx(abs(result.pearson_r), abs=1e-12)


# --- edge cases ---

def test_single_point_insufficient_data():
    result = check_spatial_correlation([1.0], [1.0])
    assert result.flag == "INSUFFICIENT_DATA"
    assert result.correlated is False


def test_empty_inputs_insufficient_data():
    result = check_spatial_correlation([], [])
    assert result.flag == "INSUFFICIENT_DATA"


def test_mismatched_lengths_uses_shorter():
    x = [float(i) for i in range(50)]
    y = [float(i) for i in range(3)]
    result = check_spatial_correlation(x, y)
    assert result.n_points == 3


# --- return type ---

def test_returns_spatial_corr_result():
    x = [float(i) for i in range(10)]
    y = [float(i) for i in range(10)]
    result = check_spatial_correlation(x, y)
    assert isinstance(result, SpatialCorrResult)


def test_result_is_frozen():
    x = [float(i) for i in range(10)]
    y = [float(i) for i in range(10)]
    result = check_spatial_correlation(x, y)
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    result = check_spatial_correlation(x, y)
    text = format_spatial_corr(result)
    assert "## Spatial Correlation Check" in text


def test_format_contains_flag():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    result = check_spatial_correlation(x, y)
    text = format_spatial_corr(result)
    assert result.flag in text
