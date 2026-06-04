"""Tests for Skills/starspot_depth_bias_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from starspot_depth_bias_estimator import (
    compute_starspot_depth_bias,
    format_starspot_depth_bias_result,
)


class TestComputeStarspotDepthBias:
    def test_ok_flag(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5300.0)
        assert r.flag == "OK"

    def test_no_spots_no_bias(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.0, 5778.0, 5300.0)
        assert abs(r.true_depth_ppm - 10000.0) < 1.0

    def test_true_depth_less_than_observed(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5300.0)
        assert r.true_depth_ppm < r.observed_depth_ppm

    def test_spot_contrast_between_zero_and_one(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5300.0)
        assert 0.0 < r.spot_contrast < 1.0

    def test_inflation_factor_greater_than_one(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5300.0)
        assert r.depth_inflation_factor > 1.0

    def test_more_spots_more_bias(self) -> None:
        r1 = compute_starspot_depth_bias(10000.0, 0.02, 5778.0, 5300.0)
        r2 = compute_starspot_depth_bias(10000.0, 0.10, 5778.0, 5300.0)
        assert r2.depth_inflation_factor > r1.depth_inflation_factor

    def test_cooler_spots_more_bias(self) -> None:
        r_warm = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5500.0)
        r_cool = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5000.0)
        assert r_cool.depth_inflation_factor > r_warm.depth_inflation_factor

    def test_invalid_depth(self) -> None:
        r = compute_starspot_depth_bias(0.0, 0.05, 5778.0, 5300.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_filling_factor_negative(self) -> None:
        r = compute_starspot_depth_bias(10000.0, -0.1, 5778.0, 5300.0)
        assert r.flag == "INVALID_FILLING_FACTOR"

    def test_invalid_filling_factor_one(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 1.0, 5778.0, 5300.0)
        assert r.flag == "INVALID_FILLING_FACTOR"

    def test_invalid_stellar_teff(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 0.0, 5300.0)
        assert r.flag == "INVALID_STELLAR_TEFF"

    def test_invalid_spot_teff_hot_as_star(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5778.0)
        assert r.flag == "INVALID_SPOT_TEFF"

    def test_format_returns_string(self) -> None:
        r = compute_starspot_depth_bias(10000.0, 0.05, 5778.0, 5300.0)
        s = format_starspot_depth_bias_result(r)
        assert isinstance(s, str)
        assert r.flag in s
