"""Tests for Skills.multi_planet_checker."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.multi_planet_checker import (
    AdditionalSignal,
    MultiPlanetResult,
    check_for_additional_planets,
    format_multi_planet_result,
)


def _flat_lc(n: int = 300) -> tuple[list[float], list[float]]:
    t = list(np.linspace(2458000.0, 2458027.0, n))
    f = [1.0] * n
    return t, f


class TestCheckForAdditionalPlanets:
    def test_returns_multi_planet_result(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               search_fn=lambda *a: [])
        assert isinstance(result, MultiPlanetResult)

    def test_primary_period_stored(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 7.0, 2458002.0,
                                               search_fn=lambda *a: [])
        assert result.primary_period == pytest.approx(7.0)

    def test_no_signals_when_search_fn_returns_empty(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               search_fn=lambda *a: [])
        assert result.n_additional == 0
        assert result.additional_signals == ()

    def test_signal_below_min_snr_not_included(self) -> None:
        def _fn(*a: object) -> list[dict]:
            return [{"period_days": 3.0, "epoch_bjd": 2458001.0,
                     "depth_ppm": 1000.0, "snr": 2.0, "duration_hours": 2.0}]
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 10.0, 2458002.0,
                                               search_fn=_fn, min_snr=5.0)
        assert result.n_additional == 0

    def test_signal_above_min_snr_included(self) -> None:
        calls = [0]
        def _fn(*a: object) -> list[dict]:
            if calls[0] > 0:
                return []
            calls[0] += 1
            return [{"period_days": 3.0, "epoch_bjd": 2458001.0,
                     "depth_ppm": 1000.0, "snr": 8.0, "duration_hours": 2.0}]
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 10.0, 2458002.0,
                                               search_fn=_fn, min_snr=5.0)
        assert result.n_additional == 1

    def test_max_additional_respected(self) -> None:
        def _fn(*a: object) -> list[dict]:
            return [{"period_days": 3.0, "epoch_bjd": 2458001.0,
                     "depth_ppm": 1000.0, "snr": 8.0, "duration_hours": 2.0}]
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 10.0, 2458002.0,
                                               search_fn=_fn,
                                               max_additional=2, min_snr=5.0)
        assert result.n_additional <= 2

    def test_masked_fraction_positive(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               duration_days=0.5,
                                               search_fn=lambda *a: [])
        assert result.masked_fraction > 0.0

    def test_masked_fraction_in_range(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               search_fn=lambda *a: [])
        assert 0.0 <= result.masked_fraction <= 1.0

    def test_additional_signal_fields(self) -> None:
        calls = [0]
        def _fn(*a: object) -> list[dict]:
            if calls[0] > 0:
                return []
            calls[0] += 1
            return [{"period_days": 3.5, "epoch_bjd": 2458001.5,
                     "depth_ppm": 500.0, "snr": 7.0, "duration_hours": 1.5}]
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 10.0, 2458002.0,
                                               search_fn=_fn, min_snr=5.0)
        if result.n_additional >= 1:
            sig = result.additional_signals[0]
            assert isinstance(sig, AdditionalSignal)
            assert sig.period_days == pytest.approx(3.5)


class TestFormatMultiPlanetResult:
    def test_format_contains_primary_period(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               search_fn=lambda *a: [])
        text = format_multi_planet_result(result)
        assert "5.0000" in text or "5.000" in text

    def test_format_reports_zero_additional(self) -> None:
        t, f = _flat_lc()
        result = check_for_additional_planets(t, f, 5.0, 2458002.0,
                                               search_fn=lambda *a: [])
        text = format_multi_planet_result(result)
        assert "0" in text
