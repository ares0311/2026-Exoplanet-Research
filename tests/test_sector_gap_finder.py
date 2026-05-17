"""Tests for Skills.sector_gap_finder."""
from __future__ import annotations

import pytest
from Skills.sector_gap_finder import GapReport, find_sector_gaps, format_gap_report


class TestFindSectorGaps:
    def test_all_covered_when_transits_within_sectors(self) -> None:
        sectors = [(2458000.0, 2458027.0)]
        report = find_sector_gaps(5.0, 2458002.0, sectors, n_periods=3)
        assert report.covered_transits == report.total_transits
        assert report.missed_transits == 0

    def test_missed_when_transits_outside_sectors(self) -> None:
        # Sector covers days 0–5; transit epoch at day 8 is outside sector
        sectors = [(2458000.0, 2458005.0)]
        report = find_sector_gaps(200.0, 2458008.0, sectors, n_periods=1)
        assert report.missed_transits == 1

    def test_gap_fraction_zero_when_all_covered(self) -> None:
        sectors = [(2458000.0, 2458100.0)]
        report = find_sector_gaps(10.0, 2458005.0, sectors, n_periods=5)
        assert report.gap_fraction == pytest.approx(0.0)

    def test_n_periods_parameter_controls_count(self) -> None:
        sectors = [(2458000.0, 2458100.0)]
        report = find_sector_gaps(10.0, 2458005.0, sectors, n_periods=4)
        assert report.total_transits == 4

    def test_coverages_list_length_matches_n_periods(self) -> None:
        sectors = [(2458000.0, 2458100.0)]
        report = find_sector_gaps(10.0, 2458005.0, sectors, n_periods=6)
        assert len(report.coverages) == 6

    def test_empty_sector_times_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            find_sector_gaps(5.0, 2458000.0, [])

    def test_non_positive_period_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            find_sector_gaps(0.0, 2458000.0, [(2458000.0, 2458027.0)])

    def test_partial_transit_detection(self) -> None:
        # Transit at edge of sector — mid is in, but full window is not
        sectors = [(2458000.0, 2458010.0)]
        # epoch at 2458009.9 with 0.2 day duration — mid inside, egress outside
        report = find_sector_gaps(100.0, 2458009.9, sectors, n_periods=1, duration_days=0.2)
        total_tracked = report.covered_transits + report.partial_transits + report.missed_transits
        assert total_tracked == report.total_transits

    def test_multiple_sectors_union(self) -> None:
        sectors = [(2458000.0, 2458010.0), (2458020.0, 2458030.0)]
        report = find_sector_gaps(20.0, 2458005.0, sectors, n_periods=2)
        assert report.total_transits == 2

    def test_default_n_periods_uses_span(self) -> None:
        sectors = [(2458000.0, 2458100.0)]
        report = find_sector_gaps(10.0, 2458005.0, sectors)
        assert report.total_transits >= 1

    def test_returns_gap_report_instance(self) -> None:
        sectors = [(2458000.0, 2458027.0)]
        report = find_sector_gaps(5.0, 2458002.0, sectors, n_periods=2)
        assert isinstance(report, GapReport)

    def test_gap_fraction_between_zero_and_one(self) -> None:
        sectors = [(2458000.0, 2458010.0)]
        report = find_sector_gaps(3.0, 2458001.0, sectors, n_periods=5)
        assert 0.0 <= report.gap_fraction <= 1.0


class TestFormatGapReport:
    def test_format_contains_gap_fraction(self) -> None:
        sectors = [(2458000.0, 2458027.0)]
        report = find_sector_gaps(5.0, 2458002.0, sectors, n_periods=3)
        text = format_gap_report(report)
        assert "Gap fraction" in text or "%" in text
