"""Tests for Skills/rv_trend_detector.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_trend_detector import (
    detect_rv_trend,
    format_rv_trend_result,
)


class TestDetectRvTrend:
    def test_flat_no_trend(self) -> None:
        times = list(range(10))
        rv = [0.0] * 10
        r = detect_rv_trend(times, rv)
        assert not r.trend_detected
        assert r.flag == "OK"

    def test_linear_trend_detected(self) -> None:
        times = list(range(20))
        # Add tiny alternating noise so rms > 0 and significance is finite
        rv = [t * 10.0 + (0.1 if t % 2 == 0 else -0.1) for t in times]
        r = detect_rv_trend(times, rv)
        assert r.trend_detected
        assert r.flag == "TREND_DETECTED"

    def test_insufficient_data(self) -> None:
        r = detect_rv_trend([0.0, 1.0], [1.0, 2.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_length_mismatch(self) -> None:
        r = detect_rv_trend([0.0, 1.0, 2.0], [1.0, 2.0])
        assert r.flag == "LENGTH_MISMATCH"

    def test_slope_sign(self) -> None:
        times = list(range(10))
        rv = [t * 5.0 for t in times]
        r = detect_rv_trend(times, rv)
        assert r.linear_slope_ms_per_day > 0

    def test_rms_nonneg(self) -> None:
        times = list(range(10))
        rv = [1.0] * 10
        r = detect_rv_trend(times, rv)
        assert r.rms_residual_ms >= 0

    def test_significance_nonneg(self) -> None:
        times = list(range(10))
        rv = [t * 2.0 for t in times]
        r = detect_rv_trend(times, rv)
        if math.isfinite(r.trend_significance):
            assert r.trend_significance >= 0

    def test_n_points(self) -> None:
        times = list(range(15))
        rv = [0.0] * 15
        r = detect_rv_trend(times, rv)
        assert r.n_points == 15

    def test_noisy_flat(self) -> None:
        import random
        random.seed(42)
        times = list(range(30))
        rv = [random.gauss(0, 1) for _ in times]
        r = detect_rv_trend(times, rv)
        assert r.flag in ("OK", "TREND_DETECTED")

    def test_quadratic_coeff_finite(self) -> None:
        times = [float(i) for i in range(10)]
        rv = [0.5 * t**2 for t in times]
        r = detect_rv_trend(times, rv)
        assert math.isfinite(r.quadratic_coeff_ms_per_day2)

    def test_format_output(self) -> None:
        times = list(range(10))
        rv = [float(t) for t in times]
        r = detect_rv_trend(times, rv)
        s = format_rv_trend_result(r)
        assert "|" in s
        assert "slope" in s.lower() or "Slope" in s

    def test_custom_threshold(self) -> None:
        times = list(range(10))
        rv = [t * 0.5 for t in times]
        r_strict = detect_rv_trend(times, rv, significance_threshold=1.0)
        r_loose = detect_rv_trend(times, rv, significance_threshold=100.0)
        assert r_strict.trend_significance == r_loose.trend_significance
