"""Tests for Skills/transit_count_predictor.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_count_predictor import (
    format_transit_count,
    predict_transit_count,
)


class TestPredictTransitCount:
    def test_basic_count_30_day_window(self) -> None:
        # epoch=0, period=5, window 0..30 → transits at 0,5,10,15,20,25,30 = 7
        r = predict_transit_count(0.0, 5.0, 0.0, 30.0)
        assert r.flag == "OK"
        assert r.n_transits == 7

    def test_flag_ok(self) -> None:
        r = predict_transit_count(0.0, 5.0, 0.0, 30.0)
        assert r.flag == "OK"

    def test_invalid_period(self) -> None:
        r = predict_transit_count(0.0, 0.0, 0.0, 30.0)
        assert r.flag == "INVALID_PERIOD"
        assert r.n_transits == 0

    def test_invalid_window(self) -> None:
        r = predict_transit_count(0.0, 5.0, 30.0, 10.0)
        assert r.flag == "INVALID_WINDOW"
        assert r.n_transits == 0

    def test_no_transits_in_window(self) -> None:
        # epoch=100.6, period=1, window 0..0.5 → first transit at 0.6, not in window
        r = predict_transit_count(100.6, 1.0, 0.0, 0.5)
        assert r.n_transits == 0

    def test_single_transit(self) -> None:
        # epoch=5, period=100, window 0..10 → 1 transit at 5
        r = predict_transit_count(5.0, 100.0, 0.0, 10.0)
        assert r.n_transits == 1

    def test_next_transit_in_window(self) -> None:
        r = predict_transit_count(0.0, 5.0, 3.0, 20.0)
        assert r.next_transit_bjd >= 3.0

    def test_window_days_stored(self) -> None:
        r = predict_transit_count(0.0, 5.0, 10.0, 40.0)
        assert abs(r.window_days - 30.0) < 1e-9

    def test_n_transits_non_negative(self) -> None:
        r = predict_transit_count(0.0, 5.0, 0.0, 30.0)
        assert r.n_transits >= 0

    def test_window_shorter_than_period(self) -> None:
        # period=10, window=5 — may have 0 or 1 transit
        r = predict_transit_count(0.0, 10.0, 0.0, 5.0)
        assert r.n_transits in (0, 1)

    def test_negative_period_invalid(self) -> None:
        r = predict_transit_count(0.0, -1.0, 0.0, 10.0)
        assert r.flag == "INVALID_PERIOD"

    def test_format_returns_string(self) -> None:
        r = predict_transit_count(0.0, 5.0, 0.0, 30.0)
        s = format_transit_count(r)
        assert isinstance(s, str)
        assert "N Transits" in s or "transits" in s.lower()
