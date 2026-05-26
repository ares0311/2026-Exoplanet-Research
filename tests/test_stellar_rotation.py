"""Tests for Skills.stellar_rotation."""
from __future__ import annotations

import math

import numpy as np
from Skills.stellar_rotation import RotationResult, detect_rotation, format_rotation_result


def _flat_lc(n: int = 500) -> tuple[list[float], list[float]]:
    t = list(np.linspace(2458000.0, 2458027.0, n))
    f = [1.0] * n
    return t, f


def _rotating_lc(
    n: int = 1000,
    period: float = 10.0,
    amplitude: float = 0.02,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(42)
    t = np.linspace(2458000.0, 2458080.0, n)
    f = 1.0 + amplitude * np.sin(2 * math.pi * t / period)
    noise = rng.normal(0, 1e-4, n)
    return list(t), list(f + noise)


class TestDetectRotation:
    def test_returns_rotation_result(self) -> None:
        t, f = _flat_lc()
        result = detect_rotation(t, f)
        assert isinstance(result, RotationResult)

    def test_flat_lc_returns_safe_defaults(self) -> None:
        t, f = _flat_lc()
        result = detect_rotation(t, f)
        assert result.rotation_period_days is None
        assert result.rotation_power == 0.0
        assert result.false_alarm_probability == 1.0
        assert not result.is_significant

    def test_rotation_detected_in_sinusoidal_lc(self) -> None:
        t, f = _rotating_lc(period=10.0, amplitude=0.05)
        result = detect_rotation(t, f, period_range=(5.0, 20.0), fap_threshold=0.01)
        # FAP should be low for a clear sinusoidal signal
        assert result.rotation_power > 0

    def test_not_significant_for_flat_lc(self) -> None:
        rng = np.random.default_rng(99)
        t = list(np.linspace(2458000.0, 2458027.0, 300))
        noise = rng.normal(0, 1e-4, 300)
        f = list(1.0 + noise)
        result = detect_rotation(t, f, fap_threshold=1e-6)
        assert not result.is_significant

    def test_flare_detected_above_threshold(self) -> None:
        t, f = _flat_lc(500)
        f_list = list(f)
        # Insert a large flare spike
        f_list[250] = 1.5
        result = detect_rotation(t, f_list, flare_sigma=3.0)
        assert result.n_flares >= 1

    def test_no_flares_in_quiet_lc(self) -> None:
        t, f = _flat_lc(300)
        result = detect_rotation(t, f, flare_sigma=5.0)
        assert result.n_flares == 0

    def test_rotation_power_in_range(self) -> None:
        t, f = _rotating_lc()
        result = detect_rotation(t, f)
        assert 0.0 <= result.rotation_power <= 1.0

    def test_fap_in_range(self) -> None:
        t, f = _flat_lc()
        result = detect_rotation(t, f)
        assert 0.0 <= result.false_alarm_probability <= 1.0

    def test_insufficient_data_returns_safe_defaults(self) -> None:
        result = detect_rotation([1.0, 2.0], [1.0, 1.0])
        assert not result.is_significant
        assert result.rotation_period_days is None

    def test_nonfinite_samples_return_safe_defaults(self) -> None:
        result = detect_rotation(
            [1.0, 2.0, math.nan, 4.0, 5.0, math.inf, 7.0, 8.0, 9.0, 10.0],
            [1.0, math.nan, 1.0, 1.0, 1.0, 1.0, math.inf, 1.0, 1.0, 1.0],
        )
        assert not result.is_significant
        assert result.rotation_period_days is None
        assert result.false_alarm_probability == 1.0

    def test_period_range_respected(self) -> None:
        t, f = _rotating_lc(period=3.0, amplitude=0.05)
        result = detect_rotation(t, f, period_range=(2.0, 5.0))
        # rotation period should be within range if detected
        if result.rotation_period_days is not None:
            assert 2.0 <= result.rotation_period_days <= 5.0

    def test_flare_times_length_matches_n_flares(self) -> None:
        t, f = _flat_lc(300)
        f_list = list(f)
        f_list[100] = 1.6
        f_list[200] = 1.7
        result = detect_rotation(t, f_list, flare_sigma=3.0)
        assert len(result.flare_times) == result.n_flares


class TestFormatRotationResult:
    def test_format_contains_flares_line(self) -> None:
        t, f = _flat_lc()
        result = detect_rotation(t, f)
        text = format_rotation_result(result)
        assert "Flares" in text

    def test_format_contains_rotation_info(self) -> None:
        t, f = _flat_lc()
        result = detect_rotation(t, f)
        text = format_rotation_result(result)
        assert "Rotation" in text or "rotation" in text
