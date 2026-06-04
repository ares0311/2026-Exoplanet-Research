"""Tests for Skills/multi_sector_noise_trend.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_sector_noise_trend import format_noise_trend_result, track_noise_trend


class TestTrackNoiseTrend:
    def test_stable_noise(self) -> None:
        r = track_noise_trend([1, 2, 3, 4, 5], [500.0, 501.0, 499.0, 500.0, 500.0])
        assert r.flag == "OK"
        assert not r.degradation_detected

    def test_degradation_detected(self) -> None:
        # Clear upward trend with noise so significance > 0
        rms = [100.0 + i * 200.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(5)]
        r = track_noise_trend([1, 2, 3, 4, 5], rms)
        assert r.degradation_detected
        assert r.flag == "DEGRADATION_DETECTED"

    def test_insufficient_sectors(self) -> None:
        r = track_noise_trend([1], [500.0])
        assert r.flag == "INSUFFICIENT_SECTORS"

    def test_length_mismatch(self) -> None:
        r = track_noise_trend([1, 2, 3], [500.0, 501.0])
        assert r.flag == "LENGTH_MISMATCH"

    def test_mean_rms_correct(self) -> None:
        r = track_noise_trend([1, 2, 3], [100.0, 200.0, 300.0])
        assert abs(r.mean_rms_ppm - 200.0) < 1.0

    def test_worst_sector_identified(self) -> None:
        r = track_noise_trend([1, 2, 3, 4], [100.0, 200.0, 800.0, 150.0])
        assert r.worst_sector == 3
        assert r.worst_rms_ppm == 800.0

    def test_trend_sign_positive(self) -> None:
        r = track_noise_trend([1, 2, 3, 4, 5], [100.0, 150.0, 200.0, 250.0, 300.0])
        assert r.rms_trend_ppm_per_sector > 0

    def test_trend_sign_negative(self) -> None:
        r = track_noise_trend([1, 2, 3, 4, 5], [300.0, 250.0, 200.0, 150.0, 100.0])
        assert r.rms_trend_ppm_per_sector < 0

    def test_n_sectors_count(self) -> None:
        r = track_noise_trend([10, 20, 30], [500.0, 510.0, 495.0])
        assert r.n_sectors == 3

    def test_custom_threshold(self) -> None:
        sectors = [1, 2, 3, 4, 5]
        # Add noise so significance is non-zero
        rms = [500.0 + i * 100.0 + (8.0 if i % 2 == 0 else -8.0) for i in range(5)]
        r_strict = track_noise_trend(sectors, rms, degradation_threshold_sigma=0.1)
        r_loose = track_noise_trend(sectors, rms, degradation_threshold_sigma=1000.0)
        assert r_strict.degradation_detected
        assert not r_loose.degradation_detected

    def test_format_output(self) -> None:
        r = track_noise_trend([1, 2, 3], [500.0, 510.0, 520.0])
        s = format_noise_trend_result(r)
        assert "|" in s
        assert "trend" in s.lower() or "rms" in s.lower()

    def test_significance_finite(self) -> None:
        r = track_noise_trend([1, 2, 3, 4], [100.0, 110.0, 120.0, 130.0])
        assert math.isfinite(r.trend_significance)
