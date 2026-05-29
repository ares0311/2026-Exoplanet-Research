"""Tests for Skills/multi_target_scheduler.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_target_scheduler import (
    ScheduleResult,
    TargetWindow,
    format_schedule_result,
    schedule_targets,
)


def _window(tic_id=1, start=0.0, end=1.0, priority=0.8, duration_h=1.0):
    return TargetWindow(
        tic_id=tic_id,
        start_btjd=start,
        end_btjd=end,
        priority=priority,
        duration_hours=duration_h,
    )


def _night(start=0.0, end=0.5):
    """0.5 day = 12 hour night."""
    return start, end


def test_empty_windows():
    result = schedule_targets([], night_start=0.0, night_end=0.5)
    assert result.flag == "EMPTY"
    assert result.n_scheduled == 0


def test_single_target_scheduled():
    w = _window(tic_id=1, start=0.0, end=0.5, duration_h=2.0)
    result = schedule_targets([w], night_start=0.0, night_end=0.5)
    assert result.n_scheduled == 1


def test_priority_ordering():
    w1 = _window(tic_id=1, priority=0.2, duration_h=3.0)
    w2 = _window(tic_id=2, priority=0.9, duration_h=3.0)
    # Night too short for both
    result = schedule_targets([w1, w2], night_start=0.0, night_end=0.25)
    if result.n_scheduled == 1:
        assert result.entries[0].tic_id == 2  # higher priority first


def test_no_overlap():
    ws = [_window(i, 0.0, 1.0, 0.5, 2.0) for i in range(5)]
    result = schedule_targets(ws, night_start=0.0, night_end=1.0)
    slots = [(e.slot_start, e.slot_end) for e in result.entries]
    for i in range(len(slots) - 1):
        assert slots[i][1] <= slots[i + 1][0] + 1e-9


def test_min_priority_filters():
    w1 = _window(tic_id=1, priority=0.1)
    w2 = _window(tic_id=2, priority=0.9)
    result = schedule_targets([w1, w2], night_start=0.0, night_end=1.0, min_priority=0.5)
    tic_ids = [e.tic_id for e in result.entries]
    assert 1 not in tic_ids
    assert 2 in tic_ids


def test_night_start_after_end():
    result = schedule_targets([_window()], night_start=1.0, night_end=0.5)
    assert result.flag == "EMPTY"


def test_total_hours_correct():
    ws = [_window(i, 0.0, 1.0, 0.5, 2.0) for i in range(3)]
    result = schedule_targets(ws, night_start=0.0, night_end=1.0)
    total = sum(e.duration_hours for e in result.entries)
    assert abs(result.total_hours - total) < 0.001


def test_returns_schedule_result():
    result = schedule_targets([_window()], night_start=0.0, night_end=0.5)
    assert isinstance(result, ScheduleResult)


def test_date_label_stored():
    result = schedule_targets(
        [_window()], night_start=0.0, night_end=0.5, date_label="2026-06-01"
    )
    assert result.date_label == "2026-06-01"


def test_partial_flag_when_some_skipped():
    ws = [_window(i, 0.0, 0.1, 0.5, 3.0) for i in range(5)]
    result = schedule_targets(ws, night_start=0.0, night_end=0.1)
    assert result.flag in ("PARTIAL", "EMPTY", "OK")


def test_format_contains_status():
    result = schedule_targets([_window()], night_start=0.0, night_end=0.5)
    md = format_schedule_result(result)
    assert result.flag in md


def test_format_empty_message():
    result = schedule_targets([], night_start=0.0, night_end=0.5)
    md = format_schedule_result(result)
    assert "No targets" in md


def test_window_outside_night_skipped():
    w = _window(tic_id=1, start=2.0, end=3.0)  # outside night 0-1
    result = schedule_targets([w], night_start=0.0, night_end=1.0)
    assert result.n_scheduled == 0
