from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from dilution_factor_estimator import (
    correct_diluted_depth,
    estimate_dilution_factor,
    format_dilution_result,
)


class TestDilutionFactorEstimator:
    def test_no_neighbours_dilution_is_one(self):
        result = estimate_dilution_factor(1000.0, [])
        assert result.dilution_factor == 1.0

    def test_one_neighbour_equal_flux_dilution_half(self):
        result = estimate_dilution_factor(500.0, [500.0])
        assert abs(result.dilution_factor - 0.5) < 1e-9

    def test_multiple_neighbours(self):
        # target=100, neighbours=[100, 100, 100] → 100/400 = 0.25
        result = estimate_dilution_factor(100.0, [100.0, 100.0, 100.0])
        assert abs(result.dilution_factor - 0.25) < 1e-9

    def test_invalid_target_flux_zero(self):
        result = estimate_dilution_factor(0.0, [500.0])
        assert result.flag == "INVALID_TARGET_FLUX"
        assert result.dilution_factor == 0.0

    def test_invalid_target_flux_negative(self):
        result = estimate_dilution_factor(-1.0, [500.0])
        assert result.flag == "INVALID_TARGET_FLUX"

    def test_depth_correction_increases_depth(self):
        # neighbour dilutes signal, so corrected depth > observed depth
        result = correct_diluted_depth(1000.0, 500.0, [500.0])
        # dilution = 0.5, corrected = 1000 / 0.5 = 2000
        assert abs(result.corrected_depth_ppm - 2000.0) < 1e-6

    def test_corrected_greater_than_observed(self):
        result = correct_diluted_depth(500.0, 200.0, [800.0])
        assert result.corrected_depth_ppm > 500.0

    def test_zero_neighbours_depth_unchanged(self):
        # no neighbours → dilution = 1.0 → corrected == observed
        result = correct_diluted_depth(1000.0, 500.0, [])
        assert abs(result.corrected_depth_ppm - 1000.0) < 1e-9

    def test_format_returns_string(self):
        result = estimate_dilution_factor(1000.0, [500.0])
        md = format_dilution_result(result)
        assert isinstance(md, str)

    def test_format_contains_dilution_factor(self):
        result = estimate_dilution_factor(1000.0, [500.0])
        md = format_dilution_result(result)
        assert "Dilution" in md or "dilution" in md.lower()

    def test_n_neighbours_field(self):
        result = estimate_dilution_factor(100.0, [50.0, 75.0, 25.0])
        assert result.n_neighbours == 3

    def test_flag_ok_for_valid(self):
        result = estimate_dilution_factor(100.0, [50.0])
        assert result.flag == "OK"

    def test_invalid_depth_flag(self):
        result = correct_diluted_depth(-100.0, 500.0, [500.0])
        assert result.flag == "INVALID_DEPTH"
