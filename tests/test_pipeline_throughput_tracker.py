"""Tests for Skills/pipeline_throughput_tracker.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from pipeline_throughput_tracker import (
    ThroughputStats,
    ThroughputTracker,
    format_throughput_stats,
)


def test_empty_stats(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    stats = tracker.stats()
    assert stats.flag == "EMPTY"
    assert stats.n_total == 0


def test_record_success(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(100, "success", 1.5)
    stats = tracker.stats()
    assert stats.n_success == 1


def test_record_error(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(100, "error", 0.5)
    stats = tracker.stats()
    assert stats.n_error == 1


def test_record_no_data(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(100, "no_data", 0.2)
    stats = tracker.stats()
    assert stats.n_no_data == 1


def test_mean_duration(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(1, "success", 2.0)
    tracker.record(2, "success", 4.0)
    stats = tracker.stats()
    assert abs(stats.mean_duration_s - 3.0) < 0.01


def test_persist_across_instances(tmp_path):
    log = tmp_path / "log.json"
    t1 = ThroughputTracker(log)
    t1.record(100, "success", 1.0)
    t2 = ThroughputTracker(log)
    assert t2.stats().n_total == 1


def test_clear(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(100, "success", 1.0)
    tracker.clear()
    assert tracker.stats().flag == "EMPTY"


def test_high_error_rate_flag(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(1, "error", 0.1)
    tracker.record(2, "error", 0.1)
    tracker.record(3, "error", 0.1)
    tracker.record(4, "success", 0.1)
    tracker.record(5, "success", 0.1)
    stats = tracker.stats()
    # error rate = 3/5 = 0.6 > 0.2
    assert stats.flag == "HIGH_ERROR_RATE"


def test_format_returns_string(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(1, "success", 1.0)
    stats = tracker.stats()
    text = format_throughput_stats(stats)
    assert isinstance(text, str)
    assert "Throughput" in text


def test_format_empty():
    stats = ThroughputStats(
        n_total=0, n_success=0, n_error=0, n_no_data=0,
        mean_duration_s=None, targets_per_hour=None, flag="EMPTY"
    )
    text = format_throughput_stats(stats)
    assert "EMPTY" in text


def test_tic_id_none(tmp_path):
    tracker = ThroughputTracker(tmp_path / "log.json")
    tracker.record(None, "success", 1.0)
    stats = tracker.stats()
    assert stats.n_success == 1
