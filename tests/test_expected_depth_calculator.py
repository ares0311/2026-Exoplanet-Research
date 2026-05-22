"""Tests for expected_depth_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from expected_depth_calculator import (
    compute_expected_depth,
    format_expected_depth_result,
)


class TestComputeExpectedDepth:
    def test_result_frozen(self):
        r = compute_expected_depth(0.1, 1.0)
        try:
            r.rp_rstar_ratio = 99.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_basic_depth(self):
        # Earth-Sun: Rp=0.00916 Rsun, R*=1.0 Rsun → depth ~84 ppm
        r = compute_expected_depth(0.00916, 1.0)
        assert r.flag == "OK"
        assert 80 < r.geometric_depth_ppm < 90

    def test_hot_jupiter(self):
        # Jupiter-like: Rp=0.1 Rsun → depth ~10000 ppm
        r = compute_expected_depth(0.1, 1.0)
        assert abs(r.geometric_depth_ppm - 10000) < 100

    def test_rp_rstar_ratio(self):
        r = compute_expected_depth(0.1, 1.0)
        assert abs(r.rp_rstar_ratio - 0.1) < 1e-4

    def test_dilution_shallows_depth(self):
        r_no_dil = compute_expected_depth(0.1, 1.0, dilution_factor=1.0)
        r_dil = compute_expected_depth(0.1, 1.0, dilution_factor=0.5)
        assert r_dil.diluted_depth_ppm < r_no_dil.diluted_depth_ppm

    def test_dilution_factor_exact(self):
        r = compute_expected_depth(0.1, 1.0, dilution_factor=0.8)
        assert abs(r.diluted_depth_ppm - r.geometric_depth_ppm * 0.8) < 1

    def test_observed_depth_ratio(self):
        r = compute_expected_depth(0.1, 1.0, observed_depth_ppm=9000.0)
        assert r.depth_ratio is not None
        assert abs(r.depth_ratio - 9000.0 / r.diluted_depth_ppm) < 0.01

    def test_no_observed_depth_ratio_none(self):
        r = compute_expected_depth(0.1, 1.0)
        assert r.depth_ratio is None
        assert r.observed_depth_ppm is None

    def test_invalid_rp_zero(self):
        r = compute_expected_depth(0.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_rstar_zero(self):
        r = compute_expected_depth(0.1, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_rp_negative(self):
        r = compute_expected_depth(-0.1, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_dilution_zero(self):
        r = compute_expected_depth(0.1, 1.0, dilution_factor=0.0)
        assert r.flag == "INVALID"

    def test_invalid_dilution_gt1(self):
        r = compute_expected_depth(0.1, 1.0, dilution_factor=1.5)
        assert r.flag == "INVALID"

    def test_format_returns_string(self):
        r = compute_expected_depth(0.1, 1.0)
        s = format_expected_depth_result(r)
        assert isinstance(s, str)
        assert "Transit Depth" in s

    def test_format_contains_flag(self):
        r = compute_expected_depth(0.0, 1.0)
        s = format_expected_depth_result(r)
        assert "INVALID" in s
