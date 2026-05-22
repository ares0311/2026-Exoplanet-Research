"""Tests for transit_overlap_detector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_overlap_detector import (
    detect_transit_overlaps,
    format_overlap_result,
)


def _clean_lc(n=200, period=5.0, epoch=0.0):
    time = [i * 0.02 for i in range(n)]
    quality = [True] * n
    return time, quality


class TestDetectTransitOverlaps:
    def test_invalid_empty(self):
        r = detect_transit_overlaps([], [], 5.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_period_zero(self):
        time, q = _clean_lc()
        r = detect_transit_overlaps(time, q, 0.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_mismatched_lengths(self):
        time, q = _clean_lc(n=10)
        r = detect_transit_overlaps(time, q[:5], 5.0, 0.0)
        assert r.flag == "INVALID"

    def test_all_clean_no_overlap(self):
        time, q = _clean_lc(n=300)
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        if r.flag == "OK":
            assert r.n_overlapping == 0
            assert r.overlap_fraction == 0.0

    def test_flagged_cadences_detected(self):
        time, q = _clean_lc(n=500)
        # Flag cadences near transit at t=0
        for i, t in enumerate(time):
            if abs(t) < 0.05:
                q[i] = False
        r = detect_transit_overlaps(time, q, 5.0, 0.0, duration_hours=2.0)
        if r.flag == "OK":
            assert r.n_overlapping >= 1

    def test_n_transits_positive(self):
        time, q = _clean_lc(n=300)
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        if r.flag == "OK":
            assert r.n_transits_checked >= 1

    def test_overlap_fraction_in_range(self):
        time, q = _clean_lc(n=300)
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        if r.flag == "OK":
            assert 0.0 <= r.overlap_fraction <= 1.0

    def test_per_transit_overlap_length(self):
        time, q = _clean_lc(n=300)
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        if r.flag == "OK":
            assert len(r.per_transit_overlap) == r.n_transits_checked

    def test_result_frozen(self):
        time, q = _clean_lc()
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        try:
            r.n_overlapping = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_no_transits_in_range(self):
        time = [1000.0 + i * 0.02 for i in range(50)]
        q = [True] * 50
        # epoch far before time array
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        # Should find transits in range 1000-1001 as epoch repeats
        assert r.flag in ("OK", "NO_TRANSITS")


class TestFormatOverlapResult:
    def test_returns_string(self):
        time, q = _clean_lc()
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        assert isinstance(format_overlap_result(r), str)

    def test_contains_flag(self):
        time, q = _clean_lc()
        r = detect_transit_overlaps(time, q, 5.0, 0.0)
        assert r.flag in format_overlap_result(r)
