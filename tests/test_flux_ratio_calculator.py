"""Tests for Skills/flux_ratio_calculator.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from flux_ratio_calculator import (
    FluxRatioResult,
    compute_flux_ratios,
    format_flux_ratio_result,
)


class TestComputeFluxRatios:
    def test_returns_flux_ratio_result(self):
        result = compute_flux_ratios(12.0, [14.0])
        assert isinstance(result, FluxRatioResult)

    def test_flag_ok_with_neighbours(self):
        result = compute_flux_ratios(12.0, [14.0])
        assert result.flag == "OK"

    def test_no_neighbours_flag(self):
        result = compute_flux_ratios(12.0, [])
        assert result.flag == "NO_NEIGHBOURS"

    def test_no_neighbours_dilution_is_one(self):
        result = compute_flux_ratios(12.0, [])
        assert result.dilution_factor == pytest.approx(1.0)

    def test_no_neighbours_contamination_is_zero(self):
        result = compute_flux_ratios(12.0, [])
        assert result.total_contamination == pytest.approx(0.0)

    def test_invalid_target_mag(self):
        result = compute_flux_ratios(float("inf"), [14.0])
        assert result.flag == "INVALID"

    def test_fainter_neighbour_has_small_flux_ratio(self):
        # neighbour 5 mag fainter → flux ratio 10^(-5*0.4) = 0.01
        result = compute_flux_ratios(10.0, [15.0])
        assert result.flux_ratios[0] == pytest.approx(0.01, rel=1e-4)

    def test_same_mag_neighbour_ratio_is_one(self):
        result = compute_flux_ratios(12.0, [12.0])
        assert result.flux_ratios[0] == pytest.approx(1.0, rel=1e-4)

    def test_brighter_neighbour_ratio_greater_than_one(self):
        result = compute_flux_ratios(14.0, [12.0])
        assert result.flux_ratios[0] > 1.0

    def test_dilution_factor_between_zero_and_one(self):
        result = compute_flux_ratios(12.0, [13.0, 14.0])
        assert 0.0 < result.dilution_factor <= 1.0

    def test_total_contamination_equals_sum_of_ratios(self):
        result = compute_flux_ratios(12.0, [13.0, 14.0])
        assert result.total_contamination == pytest.approx(sum(result.flux_ratios), rel=1e-4)

    def test_nan_neighbour_gives_zero_ratio(self):
        result = compute_flux_ratios(12.0, [float("nan")])
        assert result.flux_ratios[0] == pytest.approx(0.0)

    def test_target_mag_stored(self):
        result = compute_flux_ratios(11.5, [13.0])
        assert result.target_mag == pytest.approx(11.5)

    def test_frozen_dataclass(self):
        result = compute_flux_ratios(12.0, [13.0])
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatFluxRatioResult:
    def test_returns_string(self):
        result = compute_flux_ratios(12.0, [14.0])
        md = format_flux_ratio_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = compute_flux_ratios(12.0, [14.0])
        md = format_flux_ratio_result(result)
        assert result.flag in md

    def test_no_neighbours_format(self):
        result = compute_flux_ratios(12.0, [])
        md = format_flux_ratio_result(result)
        assert "NO_NEIGHBOURS" in md
