"""Tests for Skills/multi_band_depth_comparator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_band_depth_comparator import (
    compare_multi_band_depths,
    format_multi_band_result,
)


def test_achromatic():
    r = compare_multi_band_depths(["B", "V", "R"], [1000.0, 1005.0, 995.0], [20.0, 20.0, 20.0])
    assert r.flag == "OK"
    assert not r.is_chromatic


def test_chromatic():
    # B and V differ by >10%: max frac diff = |1000-1250|/1250 = 20%
    r = compare_multi_band_depths(["B", "V"], [1000.0, 1500.0], [20.0, 20.0])
    assert r.flag == "OK"
    assert r.is_chromatic


def test_insufficient_one_band():
    r = compare_multi_band_depths(["V"], [1000.0], [20.0])
    assert r.flag == "INSUFFICIENT"


def test_invalid_length_mismatch():
    r = compare_multi_band_depths(["B", "V"], [1000.0], [20.0, 20.0])
    assert r.flag == "INVALID"


def test_invalid_zero_error():
    r = compare_multi_band_depths(["B", "V"], [1000.0, 1000.0], [20.0, 0.0])
    assert r.flag == "INVALID"


def test_invalid_negative_depth_mean():
    # If all depths are very negative → weighted mean ≤ 0
    r = compare_multi_band_depths(["B", "V"], [-1000.0, -1000.0], [20.0, 20.0])
    assert r.flag == "INVALID"


def test_weighted_mean_depth():
    r = compare_multi_band_depths(["B", "V"], [1000.0, 2000.0], [10.0, 10.0])
    assert r.weighted_mean_depth_ppm is not None
    assert abs(r.weighted_mean_depth_ppm - 1500.0) < 1.0


def test_reference_band_smallest_error():
    r = compare_multi_band_depths(["B", "V", "R"], [1000.0, 1000.0, 1000.0], [30.0, 10.0, 20.0])
    assert r.reference_band == "V"


def test_max_fractional_difference():
    r = compare_multi_band_depths(["B", "V"], [1000.0, 1100.0], [20.0, 20.0])
    assert r.max_fractional_difference is not None
    assert r.max_fractional_difference > 0


def test_chromatic_threshold_custom():
    r = compare_multi_band_depths(["B", "V"], [1000.0, 1050.0], [20.0, 20.0],
                                   chromatic_threshold=0.01)
    assert r.is_chromatic


def test_chromatic_threshold_lenient():
    r = compare_multi_band_depths(["B", "V"], [1000.0, 1050.0], [20.0, 20.0],
                                   chromatic_threshold=0.20)
    assert not r.is_chromatic


def test_band_names_stored():
    r = compare_multi_band_depths(["g", "r", "i"], [1000.0] * 3, [20.0] * 3)
    assert r.band_names == ("g", "r", "i")


def test_format_ok():
    r = compare_multi_band_depths(["V", "R"], [1000.0, 1010.0], [20.0, 25.0])
    text = format_multi_band_result(r)
    assert "Multi-Band" in text
    assert "OK" in text


def test_format_chromatic():
    r = compare_multi_band_depths(["V", "R"], [1000.0, 1500.0], [20.0, 20.0])
    text = format_multi_band_result(r)
    assert "CHROMATIC" in text
