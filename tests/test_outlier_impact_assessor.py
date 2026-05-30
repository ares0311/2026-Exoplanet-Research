import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from outlier_impact_assessor import (
    OutlierImpactResult,
    assess_outlier_impact,
    format_outlier_impact,
)

# --- happy path ---

def test_no_outliers_clean_flux():
    flux = [1.0] * 20
    result = assess_outlier_impact(flux)
    assert result.n_outliers == 0
    assert result.outlier_indices == ()
    assert result.flag == "OK"


def test_obvious_outlier_detected():
    # Non-zero spread needed for MAD > 0; then extreme outlier is detectable
    flux = [1.0 + i * 0.001 for i in range(19)] + [10.0]
    result = assess_outlier_impact(flux, n_sigma=3.0)
    assert result.n_outliers >= 1
    assert 19 in result.outlier_indices


def test_depth_with_gt_depth_without_when_outlier_present():
    flux = [1.0] * 19 + [5.0]
    result = assess_outlier_impact(flux, n_sigma=3.0)
    assert result.depth_with_ppm >= result.depth_without_ppm


def test_depth_values_are_floats():
    flux = [0.99, 1.00, 1.01, 0.98, 1.02]
    result = assess_outlier_impact(flux)
    assert isinstance(result.depth_with_ppm, float)
    assert isinstance(result.depth_without_ppm, float)


# --- flag boundary ---

def test_flag_ok_small_depth_change():
    # uniform flux: no outliers, depth_change = 0 → OK
    flux = [1.0] * 50
    result = assess_outlier_impact(flux)
    assert result.flag == "OK"


def test_flag_outlier_impact():
    # outlier causes > 20% depth change
    base = [0.99 + i * 0.0001 for i in range(30)]
    base.append(5.0)  # extreme outlier
    result = assess_outlier_impact(base, n_sigma=3.0)
    assert result.flag == "OUTLIER_IMPACT"


def test_depth_change_pct_non_negative():
    flux = [1.0, 0.999, 1.001, 0.998, 1.002, 0.997]
    result = assess_outlier_impact(flux)
    assert result.depth_change_pct >= 0.0


# --- edge cases ---

def test_single_element_returns_ok():
    result = assess_outlier_impact([1.0])
    assert result.n_outliers == 0
    assert result.flag == "OK"


def test_empty_list_returns_ok():
    result = assess_outlier_impact([])
    assert result.n_outliers == 0
    assert result.flag == "OK"


def test_two_identical_values():
    result = assess_outlier_impact([1.0, 1.0])
    assert result.depth_with_ppm == 0.0
    assert result.n_outliers == 0


def test_custom_sigma_stricter():
    flux = [1.0] * 18 + [1.5, 1.6]
    result_strict = assess_outlier_impact(flux, n_sigma=2.0)
    result_loose = assess_outlier_impact(flux, n_sigma=5.0)
    assert result_strict.n_outliers >= result_loose.n_outliers


# --- return type ---

def test_returns_outlier_impact_result():
    result = assess_outlier_impact([1.0, 0.99, 1.01])
    assert isinstance(result, OutlierImpactResult)


def test_result_is_frozen():
    result = assess_outlier_impact([1.0, 0.99, 1.01])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = assess_outlier_impact([1.0, 0.99, 1.01])
    text = format_outlier_impact(result)
    assert "## Outlier Impact Assessment" in text


def test_format_contains_flag():
    result = assess_outlier_impact([1.0, 0.99, 1.01])
    text = format_outlier_impact(result)
    assert result.flag in text
