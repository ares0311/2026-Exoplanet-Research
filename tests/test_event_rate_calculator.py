import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from event_rate_calculator import EventRateResult, calculate_event_rate, format_event_rate

# --- happy path ---

def test_basic_many_events():
    result = calculate_event_rate(period_days=5.0, duration_hours=2.0, coverage_days=50.0)
    assert result.n_events == 10
    assert result.flag == "OK"


def test_n_events_is_integer_division():
    result = calculate_event_rate(period_days=7.0, duration_hours=3.0, coverage_days=30.0)
    assert result.n_events == int(30.0 / 7.0)


def test_duty_cycle_computed_correctly():
    result = calculate_event_rate(period_days=10.0, duration_hours=2.4, coverage_days=100.0)
    expected = 2.4 / 24.0 / 10.0
    assert result.duty_cycle == pytest.approx(expected, abs=1e-12)


def test_coverage_and_period_stored():
    result = calculate_event_rate(period_days=3.0, duration_hours=1.5, coverage_days=27.0)
    assert result.coverage_days == 27.0
    assert result.period_days == 3.0


# --- flag boundary ---

def test_flag_ok_three_or_more_events():
    result = calculate_event_rate(period_days=5.0, duration_hours=2.0, coverage_days=20.0)
    assert result.n_events >= 3
    assert result.flag == "OK"


def test_flag_few_events_when_two():
    # 2 events → FEW_EVENTS
    result = calculate_event_rate(period_days=10.0, duration_hours=2.0, coverage_days=25.0)
    assert result.n_events == 2
    assert result.flag == "FEW_EVENTS"


def test_flag_few_events_when_one():
    result = calculate_event_rate(period_days=30.0, duration_hours=3.0, coverage_days=20.0)
    assert result.n_events < 3
    assert result.flag == "FEW_EVENTS"


def test_flag_ok_exactly_three_events():
    result = calculate_event_rate(period_days=10.0, duration_hours=2.0, coverage_days=30.0)
    assert result.n_events == 3
    assert result.flag == "OK"


# --- edge cases ---

def test_invalid_zero_period():
    result = calculate_event_rate(period_days=0.0, duration_hours=2.0, coverage_days=30.0)
    assert result.flag == "INVALID_PERIOD"
    assert result.n_events == 0


def test_invalid_negative_period():
    result = calculate_event_rate(period_days=-5.0, duration_hours=2.0, coverage_days=30.0)
    assert result.flag == "INVALID_PERIOD"


def test_coverage_shorter_than_period():
    result = calculate_event_rate(period_days=100.0, duration_hours=2.0, coverage_days=50.0)
    assert result.n_events == 0
    assert result.flag == "FEW_EVENTS"


# --- return type ---

def test_returns_event_rate_result():
    result = calculate_event_rate(5.0, 2.0, 50.0)
    assert isinstance(result, EventRateResult)


def test_result_is_frozen():
    result = calculate_event_rate(5.0, 2.0, 50.0)
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = calculate_event_rate(5.0, 2.0, 50.0)
    text = format_event_rate(result)
    assert "## Transit Event Rate" in text


def test_format_contains_flag():
    result = calculate_event_rate(5.0, 2.0, 50.0)
    text = format_event_rate(result)
    assert result.flag in text
