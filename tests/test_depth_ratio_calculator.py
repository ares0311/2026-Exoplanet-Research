import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from depth_ratio_calculator import DepthRatioResult, compute_depth_ratio, format_depth_ratio


# --- happy path ---

def test_planet_like_small_secondary():
    # ratio = 50/1000 = 0.05 < 0.1 → PLANET_LIKE
    result = compute_depth_ratio(primary_depth_ppm=1000.0, secondary_depth_ppm=50.0)
    assert result.classification == "PLANET_LIKE"
    assert result.flag == "OK"
    assert result.depth_ratio == pytest.approx(0.05, abs=1e-9)


def test_eb_likely_moderate_secondary():
    # ratio = 300/1000 = 0.3 → EB_LIKELY
    result = compute_depth_ratio(primary_depth_ppm=1000.0, secondary_depth_ppm=300.0)
    assert result.classification == "EB_LIKELY"
    assert result.flag == "EB_FLAG"


def test_symmetric_eb_deep_secondary():
    # ratio = 600/1000 = 0.6 >= 0.5 → SYMMETRIC_EB
    result = compute_depth_ratio(primary_depth_ppm=1000.0, secondary_depth_ppm=600.0)
    assert result.classification == "SYMMETRIC_EB"
    assert result.flag == "EB_FLAG"


def test_depths_stored_correctly():
    result = compute_depth_ratio(2000.0, 400.0)
    assert result.primary_depth_ppm == 2000.0
    assert result.secondary_depth_ppm == 400.0


# --- flag boundary ---

def test_flag_ok_ratio_below_0_1():
    result = compute_depth_ratio(1000.0, 90.0)
    assert result.depth_ratio == pytest.approx(0.09, abs=1e-9)
    assert result.flag == "OK"


def test_flag_eb_ratio_exactly_0_1():
    # ratio = 0.1 → EB_FLAG
    result = compute_depth_ratio(1000.0, 100.0)
    assert result.depth_ratio == pytest.approx(0.1, abs=1e-9)
    assert result.flag == "EB_FLAG"
    assert result.classification == "EB_LIKELY"


def test_classification_boundary_0_5():
    # ratio = 0.5 → SYMMETRIC_EB
    result = compute_depth_ratio(1000.0, 500.0)
    assert result.classification == "SYMMETRIC_EB"


def test_classification_just_below_0_5():
    result = compute_depth_ratio(1000.0, 499.0)
    assert result.classification == "EB_LIKELY"


# --- edge cases ---

def test_zero_primary_depth_invalid():
    result = compute_depth_ratio(0.0, 100.0)
    assert result.flag == "INVALID_PRIMARY_DEPTH"
    assert result.classification == "UNDEFINED"


def test_negative_primary_depth_invalid():
    result = compute_depth_ratio(-100.0, 50.0)
    assert result.flag == "INVALID_PRIMARY_DEPTH"


def test_zero_secondary_depth():
    result = compute_depth_ratio(1000.0, 0.0)
    assert result.depth_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.classification == "PLANET_LIKE"
    assert result.flag == "OK"


# --- return type ---

def test_returns_depth_ratio_result():
    result = compute_depth_ratio(1000.0, 100.0)
    assert isinstance(result, DepthRatioResult)


def test_result_is_frozen():
    result = compute_depth_ratio(1000.0, 100.0)
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = compute_depth_ratio(1000.0, 100.0)
    text = format_depth_ratio(result)
    assert "## Depth Ratio Calculator" in text


def test_format_contains_flag():
    result = compute_depth_ratio(1000.0, 100.0)
    text = format_depth_ratio(result)
    assert result.flag in text
