"""Tests for Skills/observation_yield_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_yield_estimator import estimate_observation_yield, format_yield_result


class TestEstimateObservationYield:
    def test_basic_ok(self) -> None:
        # period=5, epoch=0, campaign 0-100 days => 21 transits
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        assert r.flag == "OK"

    def test_invalid_period(self) -> None:
        r = estimate_observation_yield(0.0, 0.0, 0.0, 100.0, 8.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_campaign_window(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 100.0, 50.0, 8.0)
        assert r.flag == "INVALID_CAMPAIGN_WINDOW"

    def test_invalid_nightly_window(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, -1.0)
        assert r.flag == "INVALID_NIGHTLY_WINDOW"

    def test_invalid_transit_duration(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0, transit_duration_hours=0.0)
        assert r.flag == "INVALID_TRANSIT_DURATION"

    def test_n_transits_positive(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        assert r.n_transits_in_campaign > 0

    def test_observable_le_total(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        assert r.n_observable <= r.n_transits_in_campaign

    def test_transit_wider_than_window_zero_observable(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 1.0, transit_duration_hours=3.0)
        assert r.n_observable == 0

    def test_campaign_days_correct(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        assert abs(r.campaign_days - 100.0) < 1e-9

    def test_observable_fraction_in_range(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        assert 0.0 <= r.observable_fraction <= 1.0

    def test_full_night_window_high_yield(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 24.0)
        assert r.n_observable == r.n_transits_in_campaign

    def test_format_returns_string(self) -> None:
        r = estimate_observation_yield(5.0, 0.0, 0.0, 100.0, 8.0)
        s = format_yield_result(r)
        assert isinstance(s, str)
        assert "Yield" in s
