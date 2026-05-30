"""Tests for Skills/follow_up_priority_tracker.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from follow_up_priority_tracker import (  # noqa: E402
    FollowUpPriorityTracker,
    PriorityRecord,
    format_priority_tracker,
)


def _tracker():
    tmp = tempfile.mktemp(suffix=".json")
    return FollowUpPriorityTracker(tmp)


def test_empty_tracker_top_priority():
    t = _tracker()
    assert t.top_priority() == []


def test_update_returns_record():
    t = _tracker()
    rec = t.update(12345, 2, "Good SNR")
    assert isinstance(rec, PriorityRecord)
    assert rec.tic_id == 12345
    assert rec.priority == 2


def test_history_empty_for_unknown():
    t = _tracker()
    t.update(11111, 1, "First")
    assert t.get_history(99999) == []


def test_history_returns_all_records():
    t = _tracker()
    t.update(12345, 2, "First update")
    t.update(12345, 1, "Elevated priority")
    hist = t.get_history(12345)
    assert len(hist) == 2


def test_top_priority_sorted():
    t = _tracker()
    t.update(111, 3, "Low")
    t.update(222, 1, "High")
    t.update(333, 2, "Medium")
    top = t.top_priority(10)
    priorities = [r.priority for r in top]
    assert priorities == sorted(priorities)


def test_top_priority_deduplicates_tic():
    t = _tracker()
    t.update(111, 3, "First")
    t.update(111, 1, "Updated")
    top = t.top_priority(10)
    tic_ids = [r.tic_id for r in top]
    assert len(tic_ids) == len(set(tic_ids))


def test_priority_clamped():
    t = _tracker()
    rec = t.update(111, 10, "Out of range")
    assert rec.priority == 5
    rec2 = t.update(222, -1, "Below range")
    assert rec2.priority == 1


def test_persistence():
    tmp = tempfile.mktemp(suffix=".json")
    t1 = FollowUpPriorityTracker(tmp)
    t1.update(555, 2, "Persisted")
    t2 = FollowUpPriorityTracker(tmp)
    assert len(t2.get_history(555)) == 1


def test_top_priority_limit():
    t = _tracker()
    for i in range(10):
        t.update(i, (i % 5) + 1, f"target {i}")
    assert len(t.top_priority(3)) <= 3


def test_updated_at_is_iso():
    t = _tracker()
    rec = t.update(12345, 1, "Test")
    # Should be parseable as ISO timestamp
    assert "T" in rec.updated_at or "-" in rec.updated_at


def test_format_returns_string():
    t = _tracker()
    t.update(12345, 1, "Top priority")
    s = format_priority_tracker(t)
    assert isinstance(s, str)


def test_format_contains_tic():
    t = _tracker()
    t.update(99999, 2, "My target")
    s = format_priority_tracker(t)
    assert "99999" in s
