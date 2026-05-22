"""Tests for Skills/exclusion_zone_calculator.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from exclusion_zone_calculator import (
    ExclusionZoneResult,
    compute_exclusion_zone,
    format_exclusion_result,
)


class TestComputeExclusionZone:
    def test_returns_exclusion_zone_result(self):
        result = compute_exclusion_zone(1000.0)
        assert isinstance(result, ExclusionZoneResult)

    def test_flag_ok_for_valid_depth(self):
        result = compute_exclusion_zone(1000.0)
        assert result.flag == "OK"

    def test_zero_depth_returns_invalid(self):
        result = compute_exclusion_zone(0.0)
        assert result.flag == "INVALID"

    def test_negative_depth_returns_invalid(self):
        result = compute_exclusion_zone(-500.0)
        assert result.flag == "INVALID"

    def test_depth_stored_correctly(self):
        result = compute_exclusion_zone(2500.0)
        assert result.depth_ppm == 2500.0

    def test_no_contrast_no_centroid_cannot_exclude(self):
        result = compute_exclusion_zone(1000.0)
        assert not result.can_exclude
        assert result.excluded_by == "none"

    def test_centroid_exclusion_when_significant(self):
        result = compute_exclusion_zone(
            1000.0,
            centroid_offset_arcsec=5.0,
            centroid_sigma=4.0,
            sigma_threshold=3.0,
        )
        assert result.can_exclude
        assert result.excluded_by == "centroid"

    def test_no_centroid_exclusion_below_threshold(self):
        result = compute_exclusion_zone(
            1000.0,
            centroid_offset_arcsec=2.0,
            centroid_sigma=2.0,
            sigma_threshold=3.0,
        )
        assert not result.can_exclude

    def test_depth_exclusion_when_contrast_too_low(self):
        # Very small contrast cannot produce large depth
        result = compute_exclusion_zone(
            500000.0,  # 50% depth
            contrast_ratio=0.0001,  # tiny contaminator
        )
        assert result.can_exclude
        assert result.excluded_by == "depth"

    def test_min_separation_computed(self):
        result = compute_exclusion_zone(1000.0, pixel_scale_arcsec=21.0)
        assert result.min_separation_arcsec is not None
        assert result.min_separation_arcsec > 0

    def test_contrast_ratio_stored(self):
        result = compute_exclusion_zone(1000.0, contrast_ratio=0.1)
        assert result.contrast_ratio == pytest.approx(0.1)

    def test_no_contrast_no_min_sep_impossible_case(self):
        result = compute_exclusion_zone(0.001)
        assert result.flag == "OK"

    def test_frozen_dataclass(self):
        result = compute_exclusion_zone(1000.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatExclusionResult:
    def test_returns_string(self):
        result = compute_exclusion_zone(1000.0)
        md = format_exclusion_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = compute_exclusion_zone(1000.0)
        md = format_exclusion_result(result)
        assert "OK" in md

    def test_invalid_result_format(self):
        result = compute_exclusion_zone(-1.0)
        md = format_exclusion_result(result)
        assert "INVALID" in md
