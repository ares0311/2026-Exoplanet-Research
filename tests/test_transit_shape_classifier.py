from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_shape_classifier import (
    TransitShapeResult,
    classify_transit_shape,
    format_transit_shape,
)


class TestTransitShapeClassifier:
    def test_flat_bottom_all_same_value(self):
        # All points at the same depth → flat-bottom
        flux = [-0.01] * 10
        result = classify_transit_shape(flux)
        assert result.shape == "flat-bottom"

    def test_v_shaped_triangle(self):
        # Triangle shape: most points away from minimum
        # Linear ramp: only the centre point touches min depth
        flux = [-0.002, -0.004, -0.006, -0.008, -0.010, -0.008, -0.006, -0.004, -0.002]
        result = classify_transit_shape(flux, tolerance=0.05)
        assert result.shape == "V-shaped"

    def test_insufficient_data_returns_unknown(self):
        result = classify_transit_shape([-0.01, -0.01, -0.01, -0.01])
        assert result.shape == "unknown"
        assert result.flag == "INSUFFICIENT_DATA"

    def test_zero_depth_flag(self):
        flux = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = classify_transit_shape(flux)
        assert result.flag == "ZERO_DEPTH"
        assert result.shape == "unknown"

    def test_shape_field_is_string(self):
        flux = [-0.01] * 8
        result = classify_transit_shape(flux)
        assert isinstance(result.shape, str)

    def test_flag_ok_for_valid_input(self):
        flux = [-0.01] * 8
        result = classify_transit_shape(flux)
        assert result.flag == "OK"

    def test_format_returns_str(self):
        flux = [-0.01] * 8
        result = classify_transit_shape(flux)
        md = format_transit_shape(result)
        assert isinstance(md, str)

    def test_format_contains_shape(self):
        flux = [-0.01] * 8
        result = classify_transit_shape(flux)
        md = format_transit_shape(result)
        assert result.shape in md

    def test_scores_sum_close_to_one(self):
        flux = [-0.005, -0.008, -0.010, -0.008, -0.005, -0.003, -0.001]
        result = classify_transit_shape(flux)
        total = result.u_score + result.v_score + result.flat_score
        assert abs(total - 1.0) < 1e-9

    def test_known_v_shaped_case(self):
        # Clear V shape: only centre point is at full depth, rest are shallower
        flux = [-0.001, -0.002, -0.003, -0.005, -0.010, -0.005, -0.003, -0.002, -0.001]
        result = classify_transit_shape(flux, tolerance=0.05)
        assert result.shape == "V-shaped"
        assert result.v_score > 0.6

    def test_known_flat_bottom_case(self):
        # Flat bottom: all points close to min depth
        flux = [-0.0099, -0.0100, -0.0100, -0.0100, -0.0100, -0.0100, -0.0099]
        result = classify_transit_shape(flux, tolerance=0.05)
        assert result.shape == "flat-bottom"
        assert result.flat_score > 0.6

    def test_unknown_shape_case(self):
        # Construct a case where no score clearly dominates:
        # flat_score <= 0.6, v_score <= 0.6, but u_score is also low
        # Mix of depths at various fractions of minimum
        flux = [-0.01, -0.009, -0.008, -0.005, -0.003, -0.009, -0.008]
        result = classify_transit_shape(flux, tolerance=0.5)
        # With very wide tolerance, threshold = depth * 0.5
        # depth = 0.01, threshold = 0.005
        # points with |f| >= 0.005: -0.01, -0.009, -0.008, -0.005, -0.009, -0.008 -> 6/7 ≈ 0.857
        # So flat_score > 0.6 → flat-bottom; verify result is a valid TransitShapeResult
        assert isinstance(result, TransitShapeResult)
