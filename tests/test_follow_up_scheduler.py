"""Tests for Skills.follow_up_scheduler."""
from __future__ import annotations

import pytest
from Skills.follow_up_scheduler import TransitEvent, build_schedule, format_schedule


def _sched(**kw: object) -> list[TransitEvent]:
    defaults: dict = {
        "period_days": 5.0,
        "epoch_bjd": 2458002.0,
        "depth_ppm": 5000.0,
        "duration_hours": 2.0,
        "window_start_bjd": 2458000.0,
        "window_end_bjd": 2458030.0,
    }
    defaults.update(kw)
    return build_schedule(**defaults)  # type: ignore[arg-type]


class TestBuildSchedule:
    def test_returns_list(self) -> None:
        events = _sched()
        assert isinstance(events, list)

    def test_transits_within_window(self) -> None:
        events = _sched(window_start_bjd=2458000.0, window_end_bjd=2458020.0)
        for e in events:
            assert 2458000.0 <= e.mid_bjd <= 2458020.0

    def test_max_events_respected(self) -> None:
        events = _sched(period_days=0.5, window_end_bjd=2458100.0, max_events=5)
        assert len(events) <= 5

    def test_priority_score_in_range(self) -> None:
        for e in _sched():
            assert 0.0 <= e.priority_score <= 1.0

    def test_depth_ppm_stored(self) -> None:
        events = _sched(depth_ppm=3000.0)
        assert all(e.depth_ppm == pytest.approx(3000.0) for e in events)

    def test_duration_stored(self) -> None:
        events = _sched(duration_hours=1.5)
        assert all(e.duration_hours == pytest.approx(1.5) for e in events)

    def test_no_events_before_window(self) -> None:
        # epoch before window; period large enough no transit falls in window
        events = build_schedule(
            200.0, 2457900.0, 5000.0, 2.0,
            window_start_bjd=2458000.0,
            window_end_bjd=2458010.0,
        )
        assert len(events) == 0

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError):
            build_schedule(-1.0, 2458002.0, 5000.0, 2.0,
                           window_start_bjd=2458000.0, window_end_bjd=2458020.0)

    def test_reversed_window_raises(self) -> None:
        with pytest.raises(ValueError):
            build_schedule(5.0, 2458002.0, 5000.0, 2.0,
                           window_start_bjd=2458030.0, window_end_bjd=2458000.0)

    def test_transit_number_set(self) -> None:
        events = _sched()
        assert all(isinstance(e.transit_number, int) for e in events)

    def test_mid_iso_contains_year(self) -> None:
        events = _sched()
        for e in events:
            # BJD 2458000 ≈ 2017-09-xx; accept any year starting with "20" or "BJD" fallback
            assert e.mid_iso[:2] == "20" or "BJD" in e.mid_iso


class TestFormatSchedule:
    def test_empty_events_message(self) -> None:
        text = format_schedule([])
        assert "No observable" in text

    def test_format_contains_tic_id(self) -> None:
        events = _sched()
        text = format_schedule(events, tic_id=12345)
        assert "12345" in text

    def test_format_contains_table_header(self) -> None:
        events = _sched()
        text = format_schedule(events)
        assert "Priority" in text
