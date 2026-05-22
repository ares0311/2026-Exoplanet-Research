"""Tests for Skills/window_function_analyzer.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from window_function_analyzer import (
    WindowFunctionResult,
    compute_window_function,
    find_alias_periods,
    format_window_result,
)


def _uniform_times(n: int = 100, dt: float = 0.5) -> list[float]:
    return [i * dt for i in range(n)]


def _gapped_times(
    n_before: int = 30, gap: float = 10.0, n_after: int = 30, dt: float = 0.5
) -> list[float]:
    t = [i * dt for i in range(n_before)]
    t_start2 = t[-1] + gap
    t += [t_start2 + i * dt for i in range(1, n_after + 1)]
    return t


class TestComputeWindowFunction:
    def test_returns_window_function_result(self):
        time = _uniform_times()
        result = compute_window_function(time)
        assert isinstance(result, WindowFunctionResult)

    def test_flag_ok_for_valid_input(self):
        time = _uniform_times()
        result = compute_window_function(time)
        assert result.flag == "OK"

    def test_empty_input_returns_invalid(self):
        result = compute_window_function([])
        assert result.flag == "INVALID"

    def test_too_few_points_returns_invalid(self):
        result = compute_window_function([0.0, 1.0])
        assert result.flag == "INVALID"

    def test_freq_and_power_same_length(self):
        time = _uniform_times()
        result = compute_window_function(time, n_freqs=200)
        assert len(result.freq_grid) == len(result.window_power)

    def test_n_freqs_respected(self):
        time = _uniform_times()
        result = compute_window_function(time, n_freqs=300)
        assert len(result.freq_grid) == 300

    def test_power_normalised_to_one(self):
        time = _uniform_times()
        result = compute_window_function(time)
        assert max(result.window_power) == pytest.approx(1.0, abs=1e-5)

    def test_duty_cycle_between_zero_and_one(self):
        time = _uniform_times()
        result = compute_window_function(time)
        assert 0.0 <= result.duty_cycle <= 1.0

    def test_invalid_freq_range_returns_invalid(self):
        time = _uniform_times()
        result = compute_window_function(time, freq_min=2.0, freq_max=1.0)
        assert result.flag == "INVALID"

    def test_gapped_time_series_flag_ok(self):
        time = _gapped_times()
        result = compute_window_function(time)
        assert result.flag == "OK"

    def test_alias_periods_are_sorted(self):
        time = _gapped_times()
        result = compute_window_function(time)
        assert list(result.alias_periods_days) == sorted(result.alias_periods_days)

    def test_frozen_dataclass(self):
        time = _uniform_times()
        result = compute_window_function(time)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFindAliasPeriods:
    def test_returns_list(self):
        time = _uniform_times()
        result = compute_window_function(time)
        aliases = find_alias_periods(result, 5.0)
        assert isinstance(aliases, list)

    def test_empty_result_returns_empty(self):
        result = WindowFunctionResult((), (), (), 0.0, "INVALID")
        aliases = find_alias_periods(result, 5.0)
        assert aliases == []

    def test_negative_period_returns_empty(self):
        time = _uniform_times()
        result = compute_window_function(time)
        aliases = find_alias_periods(result, -1.0)
        assert aliases == []

    def test_alias_within_tolerance_returned(self):
        result = WindowFunctionResult(
            freq_grid=(0.5,),
            window_power=(1.0,),
            alias_periods_days=(2.001,),
            duty_cycle=0.5,
            flag="OK",
        )
        aliases = find_alias_periods(result, 2.0, alias_threshold=0.05)
        assert 2.001 in aliases


class TestFormatWindowResult:
    def test_returns_string(self):
        time = _uniform_times()
        result = compute_window_function(time)
        md = format_window_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        time = _uniform_times()
        result = compute_window_function(time)
        md = format_window_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = compute_window_function([])
        md = format_window_result(result)
        assert "INVALID" in md
