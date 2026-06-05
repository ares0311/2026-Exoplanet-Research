"""Tests for Skills/detection_bias_corrector.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from detection_bias_corrector import compute_detection_bias, format_detection_bias_result


class TestComputeDetectionBias:
    def test_ok_flag(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert r.flag == "OK"

    def test_geometric_probability_positive(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert r.geometric_probability > 0.0

    def test_geometric_probability_le_one(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert r.geometric_probability <= 1.0

    def test_photometric_completeness_between_zero_one(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert 0.0 <= r.photometric_completeness <= 1.0

    def test_combined_efficiency_between_zero_one(self) -> None:
        r = compute_detection_bias(10.0, 4.0)  # 4 R_earth for detectable SNR
        assert 0.0 < r.combined_efficiency <= 1.0

    def test_correction_ge_one(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert r.occurrence_rate_correction >= 1.0

    def test_short_period_higher_geometric_prob(self) -> None:
        r_short = compute_detection_bias(3.0, 1.0)
        r_long = compute_detection_bias(365.0, 1.0)
        assert r_short.geometric_probability > r_long.geometric_probability

    def test_larger_planet_higher_completeness(self) -> None:
        r_small = compute_detection_bias(10.0, 0.5)
        r_large = compute_detection_bias(10.0, 10.0)
        assert r_large.photometric_completeness >= r_small.photometric_completeness

    def test_invalid_period(self) -> None:
        r = compute_detection_bias(0.0, 1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_radius(self) -> None:
        r = compute_detection_bias(10.0, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_n_expected_positive(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        assert r.n_expected_per_star > 0.0

    def test_result_frozen(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        try:
            r.geometric_probability = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_detection_bias(10.0, 1.0)
        s = format_detection_bias_result(r)
        assert isinstance(s, str)
        assert r.flag in s
