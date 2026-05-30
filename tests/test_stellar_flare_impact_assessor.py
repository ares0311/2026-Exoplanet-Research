"""Tests for Skills/stellar_flare_impact_assessor.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_flare_impact_assessor import assess_flare_impact, format_flare_impact_result


class TestAssessFlareImpact:
    def test_small_flare_ok(self) -> None:
        r = assess_flare_impact(50.0, 0.1, 2.0)
        assert r.flag == "OK"
        assert r.significant is False

    def test_large_flare_significant(self) -> None:
        r = assess_flare_impact(10000.0, 3.0, 2.0)
        assert r.significant is True
        assert r.flag == "SIGNIFICANT_CONTAMINATION"

    def test_invalid_amplitude(self) -> None:
        r = assess_flare_impact(-100.0, 1.0, 2.0)
        assert r.flag == "INVALID_AMPLITUDE"

    def test_invalid_flare_duration(self) -> None:
        r = assess_flare_impact(1000.0, 0.0, 2.0)
        assert r.flag == "INVALID_FLARE_DURATION"

    def test_invalid_transit_duration(self) -> None:
        r = assess_flare_impact(1000.0, 1.0, 0.0)
        assert r.flag == "INVALID_TRANSIT_DURATION"

    def test_invalid_baseline(self) -> None:
        r = assess_flare_impact(1000.0, 1.0, 2.0, baseline_flux=0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_overlap_fraction_max_1(self) -> None:
        r = assess_flare_impact(1000.0, 0.5, 2.0)
        assert r.overlap_fraction <= 1.0

    def test_flare_shorter_than_transit(self) -> None:
        r = assess_flare_impact(1000.0, 0.5, 2.0)
        assert r.overlap_fraction < 1.0

    def test_flare_longer_than_transit(self) -> None:
        r = assess_flare_impact(1000.0, 5.0, 2.0)
        assert abs(r.overlap_fraction - 1.0) < 1e-9

    def test_contamination_positive(self) -> None:
        r = assess_flare_impact(1000.0, 1.0, 2.0)
        assert r.depth_contamination_ppm >= 0

    def test_custom_threshold(self) -> None:
        r = assess_flare_impact(200.0, 2.0, 2.0, significance_threshold_ppm=50.0)
        assert r.significant is True

    def test_format_returns_string(self) -> None:
        r = assess_flare_impact(500.0, 1.0, 2.0)
        s = format_flare_impact_result(r)
        assert isinstance(s, str)
        assert "Flare" in s
