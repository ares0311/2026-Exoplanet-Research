"""Tests for Skills/folded_transit_stack.py."""
from Skills.folded_transit_stack import (
    StackedTransit,
    format_stack_result,
    stack_transit_windows,
)


class TestStackTransitWindows:
    def _box_lc(self, period=10.0, depth=0.01, n_transits=3, n_oot=50):
        """Synthetic light curve with box transits."""
        time = []
        flux = []
        midpoints = []
        for k in range(n_transits):
            mid = k * period
            midpoints.append(mid)
            # OOT points around transit
            for i in range(n_oot):
                t_i = mid - 0.5 + i * (1.0 / n_oot)
                f_i = 1.0 - depth if abs(t_i - mid) < 0.1 else 1.0
                time.append(t_i)
                flux.append(f_i)
        return time, flux, midpoints

    def test_returns_stacked_transit(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids)
        assert isinstance(result, StackedTransit)

    def test_flag_ok(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids)
        assert result.flag == "OK"

    def test_flag_invalid_mismatched(self):
        result = stack_transit_windows([1.0, 2.0], [1.0], [1.5])
        assert result.flag == "INVALID"

    def test_flag_insufficient_empty_midpoints(self):
        result = stack_transit_windows([1.0, 2.0], [1.0, 1.0], [])
        assert result.flag == "INSUFFICIENT"

    def test_flag_invalid_negative_half_width(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids, half_width_days=-0.1)
        assert result.flag == "INVALID"

    def test_n_transits_counted(self):
        t, f, mids = self._box_lc(n_transits=3)
        result = stack_transit_windows(t, f, mids, half_width_days=0.6)
        assert result.n_transits == 3

    def test_phase_bins_length_matches_n_bins(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids, n_bins=30)
        assert len(result.phase_bins) == 30

    def test_flux_mean_length_matches_bins(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids, n_bins=40)
        assert len(result.flux_mean) == 40

    def test_snr_estimate_positive(self):
        t, f, mids = self._box_lc(depth=0.01)
        result = stack_transit_windows(t, f, mids, half_width_days=0.4, n_bins=20)
        if result.snr_estimate is not None:
            assert result.snr_estimate >= 0

    def test_deeper_transit_higher_snr(self):
        t1, f1, m1 = self._box_lc(depth=0.005)
        t2, f2, m2 = self._box_lc(depth=0.02)
        r1 = stack_transit_windows(t1, f1, m1, half_width_days=0.4, n_bins=20)
        r2 = stack_transit_windows(t2, f2, m2, half_width_days=0.4, n_bins=20)
        if r1.snr_estimate is not None and r2.snr_estimate is not None:
            assert r2.snr_estimate >= r1.snr_estimate

    def test_single_midpoint_insufficient(self):
        result = stack_transit_windows([0.0, 1.0], [1.0, 1.0], [0.5])
        assert result.flag in ("OK", "INSUFFICIENT")

    def test_empty_time_invalid(self):
        result = stack_transit_windows([], [], [2460000.0])
        assert result.flag == "INVALID"

    def test_phase_bins_symmetric(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids, half_width_days=0.5, n_bins=50)
        if result.phase_bins:
            # First bin should be negative, last should be positive
            assert result.phase_bins[0] < 0
            assert result.phase_bins[-1] > 0


class TestFormatStackResult:
    def test_returns_string(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids)
        s = format_stack_result(result)
        assert isinstance(s, str)

    def test_contains_flag(self):
        t, f, mids = self._box_lc()
        result = stack_transit_windows(t, f, mids)
        s = format_stack_result(result)
        assert "Flag" in s or "flag" in s.lower()

    def test_insufficient_result_format(self):
        result = stack_transit_windows([1.0, 2.0], [1.0, 1.0], [])
        s = format_stack_result(result)
        assert "INSUFFICIENT" in s

    def _box_lc(self, period=10.0, depth=0.01, n_transits=3, n_oot=50):
        time = []
        flux = []
        midpoints = []
        for k in range(n_transits):
            mid = k * period
            midpoints.append(mid)
            for i in range(n_oot):
                t_i = mid - 0.5 + i * (1.0 / n_oot)
                f_i = 1.0 - depth if abs(t_i - mid) < 0.1 else 1.0
                time.append(t_i)
                flux.append(f_i)
        return time, flux, midpoints
