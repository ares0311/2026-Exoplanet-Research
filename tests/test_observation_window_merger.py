"""Tests for Skills/observation_window_merger.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_window_merger import TimeWindow, format_merged_windows, merge_windows


def test_single_window():
    r = merge_windows([(0.0, 1.0)])
    assert r.flag == "OK"
    assert r.n_merged == 1
    assert r.n_input == 1


def test_non_overlapping():
    r = merge_windows([(0.0, 1.0), (2.0, 3.0)])
    assert r.n_merged == 2
    assert r.total_duration == 2.0


def test_overlapping_merges():
    r = merge_windows([(0.0, 2.0), (1.0, 3.0)])
    assert r.n_merged == 1
    assert abs(r.total_duration - 3.0) < 1e-9


def test_adjacent_merges():
    r = merge_windows([(0.0, 1.0), (1.0, 2.0)])
    assert r.n_merged == 1
    assert abs(r.total_duration - 2.0) < 1e-9


def test_gap_threshold_merges():
    r = merge_windows([(0.0, 1.0), (1.5, 2.5)], gap_threshold_days=0.6)
    assert r.n_merged == 1


def test_gap_threshold_no_merge():
    r = merge_windows([(0.0, 1.0), (1.5, 2.5)], gap_threshold_days=0.4)
    assert r.n_merged == 2


def test_unsorted_input():
    r = merge_windows([(2.0, 3.0), (0.0, 1.0)])
    assert r.flag == "OK"
    assert r.n_merged == 2
    assert r.merged_windows[0].start == 0.0


def test_empty_input():
    r = merge_windows([])
    assert r.flag == "EMPTY"
    assert r.n_merged == 0


def test_invalid_type():
    r = merge_windows("not a list")
    assert r.flag == "INVALID"


def test_invalid_window_start_gt_end():
    r = merge_windows([(2.0, 1.0)])
    assert r.flag == "INVALID"


def test_total_duration_multiple():
    r = merge_windows([(0.0, 1.5), (2.0, 3.0), (4.0, 5.5)])
    assert abs(r.total_duration - 4.0) < 1e-7


def test_merged_window_dataclass():
    r = merge_windows([(0.0, 2.0)])
    w = r.merged_windows[0]
    assert isinstance(w, TimeWindow)
    assert w.start == 0.0
    assert w.end == 2.0
    assert w.duration == 2.0


def test_format_ok():
    r = merge_windows([(0.0, 1.0), (2.0, 3.0)])
    text = format_merged_windows(r)
    assert "Window Merger" in text
    assert "OK" in text


def test_format_empty():
    r = merge_windows([])
    text = format_merged_windows(r)
    assert "EMPTY" in text
