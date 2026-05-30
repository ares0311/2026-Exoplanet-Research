"""Tests for Skills/period_search_completeness_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_search_completeness_estimator import (
    estimate_search_completeness,
    format_completeness_result,
)


class TestEstimateSearchCompleteness:
    def test_high_snr_complete(self) -> None:
        r = estimate_search_completeness(1.0, 10000.0, 365.0, 100.0)
        assert r.completeness_fraction > 0.5
        assert r.flag == "OK"

    def test_low_snr_incomplete(self) -> None:
        r = estimate_search_completeness(100.0, 10.0, 27.0, 1000.0)
        assert r.flag == "BELOW_SNR_THRESHOLD"

    def test_invalid_period(self) -> None:
        r = estimate_search_completeness(0.0, 1000.0, 365.0, 100.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_noise(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 365.0, 0.0)
        assert r.flag == "INVALID_NOISE"

    def test_completeness_in_range(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 365.0, 200.0)
        assert 0.0 <= r.completeness_fraction <= 1.0

    def test_expected_n_transits(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 50.0, 100.0)
        assert abs(r.expected_n_transits - 10.0) < 0.01

    def test_gap_fraction_reduces_transits(self) -> None:
        r0 = estimate_search_completeness(5.0, 1000.0, 50.0, 100.0, gap_fraction=0.0)
        r1 = estimate_search_completeness(5.0, 1000.0, 50.0, 100.0, gap_fraction=0.5)
        assert r1.expected_n_transits < r0.expected_n_transits

    def test_expected_snr_positive(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 365.0, 100.0)
        assert r.expected_snr > 0

    def test_deeper_transit_higher_snr(self) -> None:
        r_shallow = estimate_search_completeness(5.0, 500.0, 100.0, 100.0)
        r_deep = estimate_search_completeness(5.0, 2000.0, 100.0, 100.0)
        assert r_deep.expected_snr > r_shallow.expected_snr

    def test_period_stored(self) -> None:
        r = estimate_search_completeness(7.3, 1000.0, 365.0, 100.0)
        assert r.period_days == 7.3

    def test_format_returns_string(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 365.0, 100.0)
        s = format_completeness_result(r)
        assert isinstance(s, str)
        assert "Completeness" in s

    def test_result_frozen(self) -> None:
        r = estimate_search_completeness(5.0, 1000.0, 365.0, 100.0)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass
