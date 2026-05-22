"""Tests for dilution_factor_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from dilution_factor_calculator import (
    compute_dilution_factor,
    format_dilution_result,
)


class TestComputeDilutionFactor:
    def test_no_contaminants(self):
        r = compute_dilution_factor([])
        assert r.flag == "NO_CONTAMINANTS"
        assert r.dilution_factor == 1.0
        assert r.n_contaminants == 0

    def test_single_contaminant(self):
        # Neighbour has same flux as target → dilution = 0.5
        r = compute_dilution_factor([1.0])
        assert r.flag == "OK"
        assert abs(r.dilution_factor - 0.5) < 1e-6
        assert r.n_contaminants == 1

    def test_two_contaminants(self):
        # Two neighbours each at 0.5x → F_total = 1 + 0.5 + 0.5 = 2 → D = 0.5
        r = compute_dilution_factor([0.5, 0.5])
        assert r.flag == "OK"
        assert abs(r.dilution_factor - 0.5) < 1e-6

    def test_small_contaminant(self):
        r = compute_dilution_factor([0.1])
        assert abs(r.dilution_factor - 1.0 / 1.1) < 1e-5

    def test_depth_correction(self):
        r = compute_dilution_factor([1.0], 1000.0)
        assert r.corrected_depth_ppm is not None
        assert abs(r.corrected_depth_ppm - 2000.0) < 0.1

    def test_no_depth_given(self):
        r = compute_dilution_factor([1.0])
        assert r.corrected_depth_ppm is None

    def test_invalid_negative_ratio(self):
        r = compute_dilution_factor([-0.1])
        assert r.flag == "INVALID"

    def test_invalid_zero_target_flux(self):
        r = compute_dilution_factor([1.0], target_flux=0.0)
        assert r.flag == "INVALID"

    def test_depth_preserved_no_contaminants(self):
        r = compute_dilution_factor([], 500.0)
        assert r.corrected_depth_ppm == 500.0

    def test_result_frozen(self):
        r = compute_dilution_factor([1.0])
        try:
            r.dilution_factor = 99.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatDilutionResult:
    def test_returns_string(self):
        r = compute_dilution_factor([0.5])
        assert isinstance(format_dilution_result(r), str)

    def test_contains_flag(self):
        r = compute_dilution_factor([0.5])
        assert r.flag in format_dilution_result(r)
