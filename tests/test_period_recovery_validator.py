"""Tests for Skills.period_recovery_validator."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.period_recovery_validator import (
    PeriodValidationResult,
    format_validation_result,
    validate_period,
)


def _box_lc(
    n: int = 400,
    period: float = 5.0,
    epoch: float = 2458002.0,
    depth: float = 0.01,
    duration: float = 0.1,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(0)
    t = list(np.linspace(2458000.0, 2458040.0, n))
    f = []
    for ti in t:
        ph = (ti - epoch) % period
        if ph > period / 2:
            ph -= period
        f.append(1.0 - depth if abs(ph) <= duration / 2 else 1.0)
    noise = rng.normal(0, 1e-4, n)
    return t, [fi + ni for fi, ni in zip(f, noise, strict=False)]


class TestValidatePeriod:
    def test_returns_period_validation_result(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        assert isinstance(result, PeriodValidationResult)

    def test_period_stored(self) -> None:
        t, f = _box_lc(period=7.0)
        result = validate_period(t, f, 7.0, 2458002.0)
        assert result.period_days == pytest.approx(7.0)

    def test_snr_at_p_positive_for_injected_signal(self) -> None:
        t, f = _box_lc(depth=0.01)
        result = validate_period(t, f, 5.0, 2458002.0)
        assert result.snr_at_p > 0.0

    def test_correct_period_confirmed(self) -> None:
        t, f = _box_lc(depth=0.02, period=5.0)
        result = validate_period(t, f, 5.0, 2458002.0, duration_days=0.1)
        assert result.is_correct_period

    def test_snr_values_non_negative(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        assert result.snr_at_p >= 0.0
        assert result.snr_at_half_p >= 0.0
        assert result.snr_at_double_p >= 0.0

    def test_confidence_non_negative(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        assert result.confidence >= 0.0

    def test_negative_period_raises(self) -> None:
        t, f = _box_lc()
        with pytest.raises(ValueError):
            validate_period(t, f, -1.0, 2458002.0)

    def test_custom_snr_fn_used(self) -> None:
        calls: list[float] = []
        def _fn(t: list, f: list, p: float, e: float, hd: float) -> float:
            calls.append(p)
            return 1.0
        t, f = _box_lc()
        validate_period(t, f, 5.0, 2458002.0, snr_fn=_fn)
        assert len(calls) == 3  # P, P/2, 2P

    def test_best_period_is_one_of_three(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        assert result.best_period in {5.0, 2.5, 10.0}


class TestFormatValidationResult:
    def test_format_contains_period(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        text = format_validation_result(result)
        assert "5.0" in text or "5.000" in text

    def test_format_contains_verdict(self) -> None:
        t, f = _box_lc()
        result = validate_period(t, f, 5.0, 2458002.0)
        text = format_validation_result(result)
        assert "CONFIRMED" in text or "SUSPECT" in text
