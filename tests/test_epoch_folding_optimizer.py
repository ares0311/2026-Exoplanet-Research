"""Tests for Skills/epoch_folding_optimizer.py."""
from Skills.epoch_folding_optimizer import (
    EpochOptResult,
    format_epoch_opt_result,
    optimize_epoch,
)


class TestOptimizeEpoch:
    def _midpoints(self, n=5, period=10.0, epoch=2460000.0):
        return [epoch + i * period for i in range(n)]

    def test_returns_result_type(self):
        mids = self._midpoints()
        result = optimize_epoch(mids, 10.0, 2460000.0)
        assert isinstance(result, EpochOptResult)

    def test_flag_ok(self):
        mids = self._midpoints()
        result = optimize_epoch(mids, 10.0, 2460000.0)
        assert result.flag == "OK"

    def test_perfect_midpoints_low_rms(self):
        mids = self._midpoints(n=6)
        result = optimize_epoch(mids, 10.0, 2460000.0)
        assert result.oc_rms_minutes < 0.01

    def test_optimised_epoch_close_to_init(self):
        mids = self._midpoints(n=5)
        result = optimize_epoch(mids, 10.0, 2460000.0)
        assert abs(result.optimised_epoch_bjd - 2460000.0) < 5.0

    def test_insufficient_midpoints(self):
        result = optimize_epoch([2460000.0], 10.0, 2460000.0)
        assert result.flag == "INSUFFICIENT"

    def test_empty_midpoints(self):
        result = optimize_epoch([], 10.0, 2460000.0)
        assert result.flag == "INSUFFICIENT"

    def test_invalid_period(self):
        result = optimize_epoch([2460000.0, 2460010.0], 0.0, 2460000.0)
        assert result.flag == "INVALID"

    def test_n_transits_stored(self):
        mids = self._midpoints(n=7)
        result = optimize_epoch(mids, 10.0, 2460000.0)
        assert result.n_transits == 7

    def test_initial_epoch_stored(self):
        mids = self._midpoints(n=5)
        result = optimize_epoch(mids, 10.0, 2460000.123)
        assert abs(result.initial_epoch_bjd - 2460000.123) < 1e-6

    def test_uncertainty_non_negative(self):
        mids = self._midpoints(n=5)
        result = optimize_epoch(mids, 10.0, 2460000.0)
        if result.epoch_uncertainty_days is not None:
            assert result.epoch_uncertainty_days >= 0

    def test_custom_n_grid(self):
        mids = self._midpoints(n=5)
        result = optimize_epoch(mids, 10.0, 2460000.0, n_grid=50)
        assert result.flag == "OK"

    def test_custom_half_width(self):
        mids = self._midpoints(n=5)
        result = optimize_epoch(mids, 10.0, 2460000.0, half_width_days=0.5)
        assert result.flag == "OK"

    def test_offset_epoch_corrected(self):
        # Midpoints perfectly at epoch=2460001.0, but we start at 2460000.0
        mids = [2460001.0 + i * 10.0 for i in range(6)]
        result = optimize_epoch(mids, 10.0, 2460000.0, half_width_days=2.0, n_grid=200)
        # Optimised epoch should be closer to 2460001.0
        assert abs(result.optimised_epoch_bjd - 2460001.0) < 0.5


class TestFormatEpochOptResult:
    def test_returns_string(self):
        mids = [2460000.0 + i * 10.0 for i in range(5)]
        result = optimize_epoch(mids, 10.0, 2460000.0)
        s = format_epoch_opt_result(result)
        assert isinstance(s, str)

    def test_contains_epoch(self):
        mids = [2460000.0 + i * 10.0 for i in range(5)]
        result = optimize_epoch(mids, 10.0, 2460000.0)
        s = format_epoch_opt_result(result)
        assert "epoch" in s.lower() or "Epoch" in s
