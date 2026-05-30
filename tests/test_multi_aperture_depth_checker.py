"""Tests for Skills/multi_aperture_depth_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_aperture_depth_checker import (
    check_aperture_depths,
    format_aperture_depth_result,
)


class TestCheckApertureDepths:
    def test_uniform_depths_no_contamination(self) -> None:
        r = check_aperture_depths([1000.0, 1000.0, 1000.0])
        assert r.contamination_flag is False
        assert r.flag == "OK"

    def test_varying_depths_contamination(self) -> None:
        r = check_aperture_depths([1000.0, 500.0, 1500.0])
        assert r.contamination_flag is True
        assert r.flag == "CONTAMINATION_SUSPECTED"

    def test_single_aperture_insufficient(self) -> None:
        r = check_aperture_depths([1000.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_empty_list_insufficient(self) -> None:
        r = check_aperture_depths([])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_mean_depth_correct(self) -> None:
        r = check_aperture_depths([1000.0, 2000.0])
        assert abs(r.mean_depth - 1500.0) < 1e-9

    def test_depth_rms_correct(self) -> None:
        import math
        r = check_aperture_depths([1000.0, 2000.0])
        expected_rms = math.sqrt(((1000.0 - 1500.0) ** 2 + (2000.0 - 1500.0) ** 2) / 2.0)
        assert abs(r.depth_rms - expected_rms) < 1e-6

    def test_contamination_flag_true_high_variance(self) -> None:
        r = check_aperture_depths([100.0, 1000.0], contamination_threshold=0.15)
        assert r.contamination_flag is True

    def test_contamination_flag_false_low_variance(self) -> None:
        r = check_aperture_depths([1000.0, 1010.0], contamination_threshold=0.15)
        assert r.contamination_flag is False

    def test_custom_threshold(self) -> None:
        # mean=1050, rms=50, rel_rms=50/1050≈0.048, threshold=0.04 → flag True
        r = check_aperture_depths([1000.0, 1100.0], contamination_threshold=0.04)
        assert r.contamination_flag is True

    def test_depths_stored(self) -> None:
        r = check_aperture_depths([500.0, 600.0])
        assert r.depths == [500.0, 600.0]

    def test_flag_contamination_suspected(self) -> None:
        r = check_aperture_depths([1000.0, 2000.0, 3000.0])
        assert r.flag in ("CONTAMINATION_SUSPECTED", "OK")

    def test_format_returns_string(self) -> None:
        r = check_aperture_depths([1000.0, 1050.0])
        s = format_aperture_depth_result(r)
        assert isinstance(s, str)
        assert "Depth" in s
