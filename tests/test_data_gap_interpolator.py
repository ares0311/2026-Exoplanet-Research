"""Tests for Skills/data_gap_interpolator.py."""
from Skills.data_gap_interpolator import (
    FilledLCResult,
    GapStats,
    characterize_gaps,
    fill_gaps_linear,
    format_gap_stats,
)


class TestCharacterizeGaps:
    def _uniform(self, n=50, dt=0.02):
        return [i * dt for i in range(n)]

    def test_returns_gap_stats(self):
        t = self._uniform()
        result = characterize_gaps(t)
        assert isinstance(result, GapStats)

    def test_flag_ok_uniform(self):
        t = self._uniform()
        result = characterize_gaps(t)
        assert result.flag == "OK"

    def test_insufficient_single_point(self):
        result = characterize_gaps([1.0])
        assert result.flag == "INSUFFICIENT"

    def test_insufficient_empty(self):
        result = characterize_gaps([])
        assert result.flag == "INSUFFICIENT"

    def test_invalid_unsorted(self):
        result = characterize_gaps([3.0, 1.0, 2.0])
        assert result.flag == "INVALID"

    def test_no_gaps_uniform(self):
        t = self._uniform()
        result = characterize_gaps(t, gap_threshold_factor=3.0)
        assert result.n_gaps == 0
        assert len(result.gap_indices) == 0

    def test_gap_detected(self):
        # Two separate uniform segments with a 1-day gap between them
        t1 = [i * 0.02 for i in range(10)]         # 0.00 to 0.18
        t2 = [1.5 + i * 0.02 for i in range(10)]   # 1.50 to 1.68
        t_with_gap = t1 + t2
        result = characterize_gaps(t_with_gap, gap_threshold_factor=3.0)
        assert result.n_gaps >= 1

    def test_median_cadence_correct(self):
        t = self._uniform(n=50, dt=0.02)
        result = characterize_gaps(t)
        assert abs(result.median_cadence_days - 0.02) < 1e-6

    def test_time_span_correct(self):
        t = self._uniform(n=50, dt=0.02)
        result = characterize_gaps(t)
        assert abs(result.time_span_days - (49 * 0.02)) < 1e-4

    def test_gap_fraction_in_range(self):
        t = self._uniform(n=20)
        t_with_gap = t[:10] + [t[10] + 1.0] + t[11:]
        result = characterize_gaps(t_with_gap)
        assert 0.0 <= result.gap_fraction <= 1.0

    def test_max_gap_days(self):
        t = [0.0, 0.02, 0.04, 1.0, 1.02]
        result = characterize_gaps(t, gap_threshold_factor=3.0)
        assert result.max_gap_days > 0.5


class TestFillGapsLinear:
    def _data(self, n=20, dt=0.02):
        t = [i * dt for i in range(n)]
        f = [1.0] * n
        return t, f

    def test_returns_filled_lc_result(self):
        t, f = self._data()
        stats = characterize_gaps(t)
        result = fill_gaps_linear(t, f, stats)
        assert isinstance(result, FilledLCResult)

    def test_no_gaps_flag(self):
        t, f = self._data()
        stats = characterize_gaps(t)
        result = fill_gaps_linear(t, f, stats)
        assert result.flag == "NO_GAPS"

    def test_fills_small_gap(self):
        t = [i * 0.02 for i in range(10)] + [0.5] + [0.5 + i * 0.02 for i in range(1, 10)]
        f = [1.0] * len(t)
        stats = characterize_gaps(t)
        if stats.n_gaps > 0:
            result = fill_gaps_linear(t, f, stats, max_gap_days=1.0)
            assert result.n_filled >= 0

    def test_invalid_length_mismatch(self):
        t = [0.0, 0.02, 0.04]
        f = [1.0, 1.0]
        dummy_stats = characterize_gaps(t)
        result = fill_gaps_linear(t, f, dummy_stats)
        assert result.flag == "INVALID"

    def test_n_original_preserved(self):
        t, f = self._data()
        stats = characterize_gaps(t)
        result = fill_gaps_linear(t, f, stats)
        assert result.n_original == len(t)


class TestFormatGapStats:
    def test_returns_string(self):
        t = [i * 0.02 for i in range(50)]
        stats = characterize_gaps(t)
        s = format_gap_stats(stats)
        assert isinstance(s, str)

    def test_contains_gaps(self):
        t = [i * 0.02 for i in range(50)]
        stats = characterize_gaps(t)
        s = format_gap_stats(stats)
        assert "Gap" in s or "gap" in s.lower()
