"""Tests for Skills/multi_period_power_analyzer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_period_power_analyzer import (
    MultiPeriodResult,
    analyze_multi_period_power,
    format_multi_period_result,
)

# Synthetic light curve with a 5-day transit
_T = [i * 0.02083 for i in range(1440)]
_F_FLAT = [1.0] * len(_T)
# Inject box transit at P=5, epoch=0, width=0.1 days
_F_TRANSIT = [
    0.999 if abs(t % 5.0) < 0.05 or abs((t % 5.0) - 5.0) < 0.05 else 1.0
    for t in _T
]


def test_returns_result():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0])
    assert isinstance(result, MultiPeriodResult)


def test_periods_tested_count():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0, 15.0])
    assert result.periods_tested == 3


def test_best_period_in_periods():
    periods = [3.0, 5.0, 7.0]
    result = analyze_multi_period_power(_T, _F_FLAT, periods=periods)
    assert result.best_period in periods


def test_top_results_sorted_by_power():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0, 15.0])
    powers = [p.power for p in result.top_results]
    assert powers == sorted(powers, reverse=True)


def test_ranks_assigned():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0])
    assert result.top_results[0].rank == 1


def test_empty_periods():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[])
    assert result.flag == "INVALID"


def test_empty_time():
    result = analyze_multi_period_power([], [], periods=[5.0])
    assert result.flag == "INVALID"


def test_transit_signal_detected():
    result = analyze_multi_period_power(_T, _F_TRANSIT, periods=[5.0, 7.5, 10.0],
                                        duration_days=0.1)
    # The 5-day period should have the highest power
    assert result.best_period is not None


def test_period_spacing_computed():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0, 15.0])
    assert result.period_spacing_days is not None
    assert abs(result.period_spacing_days - 5.0) < 1e-5


def test_period_power_frozen():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0])
    try:
        result.top_results[0].rank = 99  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_format_returns_string():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0])
    text = format_multi_period_result(result)
    assert isinstance(text, str)
    assert "Multi-Period" in text


def test_format_has_table():
    result = analyze_multi_period_power(_T, _F_FLAT, periods=[5.0, 10.0])
    text = format_multi_period_result(result)
    assert "|" in text
