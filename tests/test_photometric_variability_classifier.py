"""Tests for Skills/photometric_variability_classifier.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photometric_variability_classifier import (
    classify_variability,
    format_variability_result,
)


class TestClassifyVariability:
    def test_quiet_star(self) -> None:
        r = classify_variability(rms_ppm=100.0, period_power=0.1)
        assert r.flag == "OK"
        assert r.variability_class == "QUIET"

    def test_rotator_solar(self) -> None:
        r = classify_variability(
            rms_ppm=3000.0,
            dominant_period_days=25.0,
            period_power=0.8,
        )
        assert r.variability_class == "ROTATOR"

    def test_binary_high_amplitude(self) -> None:
        r = classify_variability(
            rms_ppm=60000.0,
            dominant_period_days=3.0,
            period_power=0.9,
        )
        assert r.variability_class == "BINARY"

    def test_pulsator_fast_period(self) -> None:
        r = classify_variability(
            rms_ppm=40000.0,
            dominant_period_days=0.3,
            period_power=0.95,
            peak_to_peak_ppm=80000.0,
        )
        assert r.variability_class == "PULSATOR"

    def test_invalid_rms(self) -> None:
        r = classify_variability(rms_ppm=-1.0)
        assert r.flag == "INVALID_RMS"

    def test_confidence_in_range(self) -> None:
        r = classify_variability(rms_ppm=200.0, period_power=0.2)
        assert 0.0 <= r.confidence <= 1.0

    def test_no_period_unclassified(self) -> None:
        r = classify_variability(rms_ppm=2000.0)
        assert r.flag == "OK"
        # Without period info, moderate amplitude is ambiguous
        assert r.variability_class in ("ROTATOR", "UNCLASSIFIED")

    def test_amplitude_set_from_rms(self) -> None:
        r = classify_variability(rms_ppm=200.0)
        assert r.amplitude_ppm == pytest.approx(400.0)

    def test_peak_to_peak_override(self) -> None:
        r = classify_variability(rms_ppm=100.0, peak_to_peak_ppm=1000.0, period_power=0.2)
        assert r.amplitude_ppm == 1000.0

    def test_format_output(self) -> None:
        r = classify_variability(rms_ppm=1000.0, dominant_period_days=20.0, period_power=0.7)
        s = format_variability_result(r)
        assert "|" in s
        assert "class" in s.lower()

    def test_binary_no_period_high_amp(self) -> None:
        r = classify_variability(rms_ppm=30000.0, peak_to_peak_ppm=80000.0)
        assert r.variability_class == "BINARY"

    def test_quiet_flag_ok(self) -> None:
        r = classify_variability(rms_ppm=50.0)
        assert r.flag == "OK"
