"""Tests for Skills.ephemeris_predictor."""
from __future__ import annotations

import pytest
from Skills.ephemeris_predictor import format_transit_table, predict_transits


class TestPredictTransits:
    def test_returns_n_windows(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=5)
        assert len(windows) == 5

    def test_mid_bjd_equals_epoch_plus_k_times_period(self) -> None:
        period = 7.0
        epoch = 2458000.0
        windows = predict_transits(period, epoch, n=3, t_start=epoch)
        for w in windows:
            expected = epoch + w.transit_number * period
            assert abs(w.mid_bjd - expected) < 1e-9

    def test_t_start_shifts_first_window_at_or_after(self) -> None:
        epoch = 2458000.0
        t_start = 2458100.0
        windows = predict_transits(10.0, epoch, n=3, t_start=t_start)
        assert windows[0].mid_bjd >= t_start

    def test_window_centered_on_mid(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=1, window_pad_hours=1.0)
        w = windows[0]
        center = (w.window_start + w.window_end) / 2.0
        assert abs(center - w.mid_bjd) < 1e-9

    def test_uncertainty_grows_with_transit_number_when_period_err_given(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=5, period_err=0.01, epoch_err=0.0)
        assert windows[4].uncertainty_hours >= windows[0].uncertainty_hours

    def test_zero_uncertainty_when_no_errors(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=3)
        for w in windows:
            assert w.uncertainty_hours == 0.0

    def test_negative_period_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            predict_transits(-1.0, 2458000.0)

    def test_zero_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            predict_transits(10.0, 2458000.0, n=0)

    def test_window_pad_hours_increases_window_width(self) -> None:
        ws1 = predict_transits(10.0, 2458000.0, n=1, window_pad_hours=1.0)
        ws2 = predict_transits(10.0, 2458000.0, n=1, window_pad_hours=2.0)
        w1 = ws1[0].window_end - ws1[0].window_start
        w2 = ws2[0].window_end - ws2[0].window_start
        assert w2 > w1

    def test_transit_numbers_monotonically_increasing(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=5)
        nums = [w.transit_number for w in windows]
        assert nums == sorted(nums)

    def test_epoch_err_contributes_to_uncertainty(self) -> None:
        w_no = predict_transits(10.0, 2458000.0, n=1)[0]
        w_err = predict_transits(10.0, 2458000.0, n=1, epoch_err=0.05)[0]
        assert w_err.uncertainty_hours > w_no.uncertainty_hours


class TestFormatTransitTable:
    def test_empty_list_returns_no_transits_message(self) -> None:
        result = format_transit_table([])
        assert "No transits" in result

    def test_table_contains_mid_bjd_header(self) -> None:
        windows = predict_transits(10.0, 2458000.0, n=2)
        table = format_transit_table(windows)
        assert "Mid BJD" in table
