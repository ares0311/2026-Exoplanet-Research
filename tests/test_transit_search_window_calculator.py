"""Tests for Skills/transit_search_window_calculator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_search_window_calculator import (
    compute_search_window,
    format_search_window_result,
)


class TestTransitSearchWindowCalculator:
    def test_basic_ok(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001)
        assert r.flag == "OK"
        assert len(r.windows) == 5

    def test_window_count_matches_n_epochs(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001, n_future_epochs=3)
        assert len(r.windows) == 3

    def test_predicted_time_linear(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001)
        assert abs(r.windows[0].predicted_time_bjd - 2458010.0) < 0.01
        assert abs(r.windows[1].predicted_time_bjd - 2458020.0) < 0.01

    def test_window_grows_with_epoch(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.01)
        assert r.windows[1].window_half_width_hours > r.windows[0].window_half_width_hours

    def test_sigma_total_grows_with_epoch(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.01)
        assert r.windows[2].sigma_total_hours > r.windows[0].sigma_total_hours

    def test_window_start_lt_end(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001)
        for w in r.windows:
            assert w.window_start_bjd < w.window_end_bjd

    def test_n_sigma_stored(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001, n_sigma=2.0)
        assert abs(r.n_sigma - 2.0) < 1e-9

    def test_sigma_epoch_contributes_to_window(self) -> None:
        r0 = compute_search_window(2458000.0, 10.0, 0.001, sigma_epoch_days=0.0)
        r1 = compute_search_window(2458000.0, 10.0, 0.001, sigma_epoch_days=0.1)
        assert r1.windows[0].sigma_total_hours > r0.windows[0].sigma_total_hours

    def test_invalid_period(self) -> None:
        r = compute_search_window(2458000.0, 0.0, 0.001)
        assert r.flag == "INVALID_PERIOD"
        assert len(r.windows) == 0

    def test_invalid_sigma_period(self) -> None:
        r = compute_search_window(2458000.0, 10.0, -0.001)
        assert r.flag == "INVALID_SIGMA_PERIOD"

    def test_invalid_n_epochs(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001, n_future_epochs=0)
        assert r.flag == "INVALID_N_EPOCHS"

    def test_result_is_frozen(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001)
        try:
            r.n_sigma = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = compute_search_window(2458000.0, 10.0, 0.001)
        s = format_search_window_result(r)
        assert "Epoch" in s or "epoch" in s
        assert "BJD" in s or "bjd" in s.lower()
