"""Tests for Skills/crowding_metric_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from crowding_metric_calculator import (
    CrowdingResult,
    compute_crowding_metric,
    format_crowding_result,
)


class TestComputeCrowdingMetric:
    def test_no_neighbors_crowding_one(self):
        result = compute_crowding_metric(12.0, [], [])
        assert result.flag == "OK"
        assert result.crowding_metric == 1.0

    def test_distant_neighbor_minimal_contamination(self):
        result = compute_crowding_metric(12.0, [15.0], [200.0])
        assert result.crowding_metric > 0.99

    def test_close_bright_neighbor_reduces_crowding(self):
        result = compute_crowding_metric(12.0, [12.0], [5.0])
        assert result.crowding_metric < 1.0

    def test_contamination_ratio_complement(self):
        result = compute_crowding_metric(12.0, [13.0], [10.0])
        assert abs(result.crowding_metric + result.contamination_ratio - 1.0) < 1e-9

    def test_mismatched_lengths_invalid(self):
        result = compute_crowding_metric(12.0, [13.0, 14.0], [10.0])
        assert result.flag == "INVALID"

    def test_zero_aperture_invalid(self):
        result = compute_crowding_metric(12.0, [], [], aperture_radius_arcsec=0.0)
        assert result.flag == "INVALID"

    def test_zero_psf_invalid(self):
        result = compute_crowding_metric(12.0, [], [], psf_fwhm_arcsec=0.0)
        assert result.flag == "INVALID"

    def test_n_neighbors_stored(self):
        result = compute_crowding_metric(12.0, [13.0, 14.0], [10.0, 30.0])
        assert result.n_neighbors == 2

    def test_target_flux_positive(self):
        result = compute_crowding_metric(12.0, [], [])
        assert result.target_flux > 0.0

    def test_total_flux_ge_target(self):
        result = compute_crowding_metric(12.0, [13.0], [5.0])
        assert result.total_flux >= result.target_flux

    def test_dilution_per_neighbor_not_none_with_neighbors(self):
        result = compute_crowding_metric(12.0, [14.0], [10.0])
        assert result.dilution_ppm_per_neighbor is not None

    def test_dilution_none_with_no_neighbors(self):
        result = compute_crowding_metric(12.0, [], [])
        assert result.dilution_ppm_per_neighbor is None

    def test_result_frozen(self):
        result = compute_crowding_metric(12.0, [], [])
        try:
            result.crowding_metric = 0.5
            assert False
        except Exception:
            pass

    def test_format_returns_string(self):
        result = compute_crowding_metric(12.0, [], [])
        text = format_crowding_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        result = compute_crowding_metric(12.0, [], [])
        text = format_crowding_result(result)
        assert result.flag in text
