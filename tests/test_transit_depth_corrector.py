"""Tests for Skills.transit_depth_corrector."""
from __future__ import annotations

import pytest
from Skills.transit_depth_corrector import (
    DepthCorrectionResult,
    correct_transit_depth,
    format_depth_correction,
)


class TestCorrectTransitDepth:
    def test_returns_result(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.5)
        assert isinstance(r, DepthCorrectionResult)

    def test_no_contamination_no_correction(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.0)
        assert r.correction_applied is False
        assert r.corrected_depth_ppm == pytest.approx(1000.0)

    def test_contamination_deepens_depth(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=1.0)
        # dilution = 0.5, corrected = 2000
        assert r.corrected_depth_ppm == pytest.approx(2000.0, rel=1e-4)

    def test_crowdsap_half(self) -> None:
        r = correct_transit_depth(500.0, crowdsap=0.5)
        assert r.corrected_depth_ppm == pytest.approx(1000.0, rel=1e-4)
        assert r.correction_applied is True

    def test_crowdsap_one_no_correction(self) -> None:
        r = correct_transit_depth(500.0, crowdsap=1.0)
        assert r.correction_applied is False

    def test_dilution_factor_range(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.5)
        assert 0.0 < r.dilution_factor <= 1.0

    def test_no_args_no_correction(self) -> None:
        r = correct_transit_depth(1000.0)
        assert r.correction_applied is False

    def test_measured_depth_preserved(self) -> None:
        r = correct_transit_depth(777.0, contamination_ratio=0.3)
        assert r.measured_depth_ppm == pytest.approx(777.0)

    def test_contamination_ratio_stored(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.25)
        assert r.contamination_ratio == pytest.approx(0.25, rel=1e-4)


class TestFormatDepthCorrection:
    def test_returns_string(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.5)
        assert isinstance(format_depth_correction(r), str)

    def test_measured_depth_in_output(self) -> None:
        r = correct_transit_depth(1000.0, contamination_ratio=0.5)
        assert "1000" in format_depth_correction(r)

    def test_no_correction_message(self) -> None:
        r = correct_transit_depth(1000.0)
        assert "not applied" in format_depth_correction(r)
