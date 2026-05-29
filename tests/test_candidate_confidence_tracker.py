"""Tests for Skills/candidate_confidence_tracker.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_confidence_tracker import (
    CandidateConfidenceTracker,
    ConfidenceSnapshot,
    format_confidence_trend,
)


def _tracker(tmp_path):
    return CandidateConfidenceTracker(tmp_path / "tracker.json")


def _row(fpp=0.10, dc=0.80, pathway="tfop_ready", scorer="bayesian"):
    return {
        "false_positive_probability": fpp,
        "detection_confidence": dc,
        "pathway": pathway,
        "meta": {"scorer": scorer},
    }


def test_record_returns_snapshot(tmp_path):
    tracker = _tracker(tmp_path)
    snap = tracker.record(12345, 3.0, _row())
    assert isinstance(snap, ConfidenceSnapshot)


def test_trend_none_before_record(tmp_path):
    tracker = _tracker(tmp_path)
    assert tracker.trend(12345, 3.0) is None


def test_single_run_trend(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row())
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.trend == "SINGLE_RUN"


def test_improving_trend(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row(fpp=0.80))
    tracker.record(12345, 3.0, _row(fpp=0.10))
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.trend == "IMPROVING"


def test_degrading_trend(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row(fpp=0.10))
    tracker.record(12345, 3.0, _row(fpp=0.80))
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.trend == "DEGRADING"
    assert t.flag == "DEGRADING"


def test_stable_trend(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row(fpp=0.30))
    tracker.record(12345, 3.0, _row(fpp=0.31))
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.trend == "STABLE"


def test_n_runs_count(tmp_path):
    tracker = _tracker(tmp_path)
    for _ in range(4):
        tracker.record(12345, 3.0, _row())
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.n_runs == 4


def test_delta_fpp_correct(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row(fpp=0.50))
    tracker.record(12345, 3.0, _row(fpp=0.20))
    t = tracker.trend(12345, 3.0)
    assert t is not None
    assert t.delta_fpp == pytest.approx(-0.30, abs=0.001)


def test_all_trends_returns_list(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(11111, 3.0, _row())
    tracker.record(22222, 5.0, _row())
    trends = tracker.all_trends()
    assert len(trends) == 2


def test_persistence(tmp_path):
    path = tmp_path / "t.json"
    t1 = CandidateConfidenceTracker(path)
    t1.record(12345, 3.0, _row())
    t2 = CandidateConfidenceTracker(path)
    assert t2.trend(12345, 3.0) is not None


def test_format_contains_tic(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(99999, 3.0, _row())
    t = tracker.trend(99999, 3.0)
    md = format_confidence_trend(t)
    assert "99999" in md


def test_flag_ok_for_improving(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(12345, 3.0, _row(fpp=0.80))
    tracker.record(12345, 3.0, _row(fpp=0.10))
    t = tracker.trend(12345, 3.0)
    assert t.flag == "OK"
