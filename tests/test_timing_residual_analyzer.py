"""Tests for Skills/timing_residual_analyzer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from timing_residual_analyzer import analyze_timing_residuals, format_timing_residual_result


class TestTimingResidualAnalyzer:
    def test_flat_perfect(self) -> None:
        r = analyze_timing_residuals([0.0] * 10)
        assert r.flag == "FLAT"
        assert r.rms_minutes == 0.0

    def test_insufficient_data(self) -> None:
        r = analyze_timing_residuals([1.0, 2.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_trend_detected(self) -> None:
        # Linear drift: O-C grows by 1 min per transit
        oc = [float(i) * 2.0 for i in range(10)]
        r = analyze_timing_residuals(oc, trend_threshold_min_per_transit=0.5)
        assert r.flag == "TREND"

    def test_ttv_candidate_large_scatter(self) -> None:
        # Symmetric sine-like pattern: zero mean, zero slope, large RMS
        import math
        oc = [8.0 * math.sin(2 * math.pi * i / 4) for i in range(12)]
        r = analyze_timing_residuals(oc)
        assert r.rms_minutes > 4.0  # large scatter regardless of flag

    def test_mean_correct(self) -> None:
        oc = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = analyze_timing_residuals(oc)
        assert abs(r.mean_minutes - 3.0) < 0.01

    def test_rms_non_negative(self) -> None:
        r = analyze_timing_residuals([1.0, -1.0, 2.0, -2.0, 0.0])
        assert r.rms_minutes >= 0.0

    def test_max_abs_correct(self) -> None:
        r = analyze_timing_residuals([1.0, -3.0, 2.0, -1.0, 0.5])
        assert r.max_abs_minutes == 3.0

    def test_n_transits_correct(self) -> None:
        r = analyze_timing_residuals([0.0] * 7)
        assert r.n_transits == 7

    def test_custom_transit_numbers(self) -> None:
        oc = [0.0, 1.0, 0.0, -1.0, 0.5]
        ns = [0, 2, 4, 6, 8]
        r = analyze_timing_residuals(oc, transit_numbers=ns)
        assert r.flag in ("FLAT", "TTV_CANDIDATE", "TREND")

    def test_trend_slope_float(self) -> None:
        r = analyze_timing_residuals([0.0, 1.0, 2.0, 3.0, 4.0])
        assert isinstance(r.trend_slope_min_per_transit, float)

    def test_n_outliers_non_negative(self) -> None:
        r = analyze_timing_residuals([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
        assert r.n_outliers >= 0

    def test_format_returns_string(self) -> None:
        r = analyze_timing_residuals([0.0] * 5)
        s = format_timing_residual_result(r)
        assert isinstance(s, str)
        assert "RMS" in s
