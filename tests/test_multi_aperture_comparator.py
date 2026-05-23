"""Tests for Skills/multi_aperture_comparator.py."""
import pytest
from Skills.multi_aperture_comparator import (
    ApertureCompareResult,
    compare_apertures,
    format_aperture_compare_result,
)


class TestCompareApertures:
    def test_returns_result_type(self):
        a = {"depth_ppm": 1000.0, "rms_ppm": 100.0}
        b = {"depth_ppm": 1000.0, "rms_ppm": 100.0}
        result = compare_apertures(a, b)
        assert isinstance(result, ApertureCompareResult)

    def test_flag_ok(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 1000.0}
        result = compare_apertures(a, b)
        assert result.flag == "OK"

    def test_identical_depths_no_discrepancy(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 1000.0}
        result = compare_apertures(a, b)
        assert result.depth_discrepancy_frac == 0.0
        assert not result.depth_discrepant

    def test_depth_discrepancy_computed(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 1200.0}
        result = compare_apertures(a, b)
        expected = abs(1000 - 1200) / 1200
        assert abs(result.depth_discrepancy_frac - expected) < 1e-4

    def test_depth_discrepant_above_threshold(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 2000.0}
        result = compare_apertures(a, b, depth_tol_frac=0.10)
        assert result.depth_discrepant

    def test_rms_ratio_computed(self):
        a = {"rms_ppm": 100.0}
        b = {"rms_ppm": 150.0}
        result = compare_apertures(a, b)
        assert abs(result.rms_ratio - 1.5) < 1e-4

    def test_rms_discrepant_above_threshold(self):
        a = {"rms_ppm": 100.0}
        b = {"rms_ppm": 200.0}
        result = compare_apertures(a, b, rms_tol_frac=0.20)
        assert result.rms_discrepant

    def test_scatter_ratio(self):
        a = {"depth_ppm": 1000.0, "scatter_ppm": 80.0}
        b = {"depth_ppm": 1000.0, "scatter_ppm": 160.0}
        result = compare_apertures(a, b)
        assert result.scatter_ratio is not None
        assert abs(result.scatter_ratio - 2.0) < 1e-4

    def test_contamination_index_range(self):
        a = {"depth_ppm": 1000.0, "rms_ppm": 100.0}
        b = {"depth_ppm": 2000.0, "rms_ppm": 200.0}
        result = compare_apertures(a, b)
        assert result.contamination_index is not None
        assert 0.0 <= result.contamination_index <= 2.0

    def test_non_dict_invalid(self):
        result = compare_apertures("bad", {"depth_ppm": 1000.0})
        assert result.flag == "INVALID"

    def test_empty_dicts_invalid(self):
        result = compare_apertures({}, {})
        assert result.flag == "INVALID"

    def test_missing_depth_no_discrepancy(self):
        a = {"rms_ppm": 100.0}
        b = {"rms_ppm": 150.0}
        result = compare_apertures(a, b)
        assert result.depth_discrepancy_frac is None

    def test_rms_ratio_none_when_missing(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 1100.0}
        result = compare_apertures(a, b)
        assert result.rms_ratio is None


class TestFormatApertureCompareResult:
    def test_returns_string(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 1100.0}
        result = compare_apertures(a, b)
        s = format_aperture_compare_result(result)
        assert isinstance(s, str)

    def test_contains_contamination(self):
        a = {"depth_ppm": 1000.0}
        b = {"depth_ppm": 2000.0}
        result = compare_apertures(a, b)
        s = format_aperture_compare_result(result)
        assert "Contamination" in s or "contamination" in s.lower()
