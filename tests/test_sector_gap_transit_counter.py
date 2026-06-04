"""Tests for Skills/sector_gap_transit_counter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sector_gap_transit_counter import MissedTransitResult, count_missed_transits


class TestSectorGapTransitCounter:
    def test_basic_ok(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert r.flag == "OK"

    def test_total_transits_positive(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert r.n_transits_total > 0

    def test_missed_lte_total(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert r.n_transits_missed <= r.n_transits_total

    def test_observed_plus_missed_equals_total(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert r.n_transits_observed + r.n_transits_missed == r.n_transits_total

    def test_invalid_period(self) -> None:
        r = count_missed_transits(period_days=0.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_baseline_reversed(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458300.0, baseline_end_bjd=2458000.0)
        assert r.flag == "INVALID_BASELINE"

    def test_no_gaps_all_observed(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458100.0,
            gap_windows=[])
        assert r.n_transits_missed == 0

    def test_explicit_gap_causes_miss(self) -> None:
        epoch = 2458050.0
        r = count_missed_transits(period_days=10.0, epoch_bjd=epoch,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458100.0,
            gap_windows=[(2458049.0, 2458051.0)])
        assert r.n_transits_missed >= 1

    def test_coverage_fraction_range(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert 0.0 <= r.fraction_missed <= 1.0

    def test_result_frozen(self) -> None:
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        assert isinstance(r, MissedTransitResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from sector_gap_transit_counter import format_missed_transit_result
        r = count_missed_transits(period_days=10.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0)
        s = format_missed_transit_result(r)
        assert "|" in s

    def test_long_period_few_transits(self) -> None:
        r = count_missed_transits(period_days=100.0, epoch_bjd=2458000.0,
            baseline_start_bjd=2458000.0, baseline_end_bjd=2458300.0,
            gap_windows=[])
        assert r.n_transits_total <= 4
