"""Tests for Skills/lightcurve_segment_extractor.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from lightcurve_segment_extractor import (
    extract_transit_segments,
    format_segment_summary,
)

_TIME = [i * 0.02083 for i in range(1440)]  # 30-min cadence over 30 days
_FLUX = [1.0] * len(_TIME)


def test_returns_list():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[5.0],
                                        half_window_days=0.5)
    assert isinstance(segments, list)


def test_one_segment_per_mid_time():
    segments = extract_transit_segments(_TIME, _FLUX,
                                        mid_times=[5.0, 10.0, 15.0],
                                        half_window_days=0.5)
    assert len(segments) == 3


def test_segment_has_points():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[15.0],
                                        half_window_days=0.5)
    assert segments[0].n_points > 0


def test_index_assigned():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[5.0, 10.0],
                                        half_window_days=0.5)
    assert segments[0].index == 0
    assert segments[1].index == 1


def test_time_within_window():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[10.0],
                                        half_window_days=0.5)
    for t in segments[0].time:
        assert 9.5 <= t <= 10.5


def test_empty_segment_outside_range():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[999.0],
                                        half_window_days=0.5)
    assert segments[0].flag == "EMPTY"
    assert segments[0].n_points == 0


def test_with_flux_err():
    errs = [0.001] * len(_TIME)
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[5.0],
                                        half_window_days=0.5, flux_err=errs)
    assert all(e is not None for e in segments[0].flux_err)


def test_coverage_fraction_bounded():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[15.0],
                                        half_window_days=0.5)
    assert 0.0 <= segments[0].coverage_fraction <= 1.0


def test_no_mid_times():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[])
    assert segments == []


def test_segment_frozen():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[5.0])
    try:
        segments[0].flag = "BAD"  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_format_returns_string():
    segments = extract_transit_segments(_TIME, _FLUX, mid_times=[5.0, 15.0])
    text = format_segment_summary(segments)
    assert isinstance(text, str)
    assert "Segment" in text


def test_format_empty():
    text = format_segment_summary([])
    assert "No segments" in text
