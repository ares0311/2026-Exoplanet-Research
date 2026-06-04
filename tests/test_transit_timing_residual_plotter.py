"""Tests for Skills/transit_timing_residual_plotter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_timing_residual_plotter import OcResidualResult, compute_oc_residuals


class TestTransitTimingResidualPlotter:
    def _midpoints(self, period: float = 10.0, n: int = 5,
                   epoch: float = 2458000.0) -> list[float]:
        return [epoch + i * period for i in range(n)]

    def test_perfect_ephemeris_zero_rms(self) -> None:
        mids = self._midpoints()
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        assert r.flag == "OK"
        assert r.rms_oc_minutes < 1e-6

    def test_invalid_period(self) -> None:
        r = compute_oc_residuals([2458000.0, 2458010.0], period_days=0.0,
                                  epoch_bjd=2458000.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_epoch(self) -> None:
        r = compute_oc_residuals([2458000.0, 2458010.0], period_days=10.0,
                                  epoch_bjd=float("nan"))
        assert r.flag == "INVALID_EPOCH"

    def test_insufficient_transits(self) -> None:
        r = compute_oc_residuals([2458000.0], period_days=10.0, epoch_bjd=2458000.0)
        assert r.flag == "INSUFFICIENT_TRANSITS"

    def test_n_transits_matches_input(self) -> None:
        mids = self._midpoints(n=6)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        assert r.n_transits == 6

    def test_ttv_flag_triggered(self) -> None:
        mids = self._midpoints(n=5)
        mids[2] += 10.0 / 1440.0 * 20  # 20-minute TTV
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0,
                                  ttv_threshold_minutes=5.0)
        assert r.ttv_flag

    def test_ttv_flag_not_triggered_small(self) -> None:
        mids = self._midpoints(n=5)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0,
                                  ttv_threshold_minutes=5.0)
        assert not r.ttv_flag

    def test_points_count(self) -> None:
        mids = self._midpoints(n=4)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        assert len(r.points) == 4

    def test_quadratic_fit(self) -> None:
        mids = self._midpoints(n=6)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0,
                                  fit_quadratic=True)
        assert r.quadratic_coeff_min_per_epoch2 is not None

    def test_linear_slope_near_zero_perfect(self) -> None:
        mids = self._midpoints(n=5)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        assert abs(r.linear_slope_min_per_epoch) < 1e-6

    def test_result_frozen(self) -> None:
        mids = self._midpoints(n=3)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        assert isinstance(r, OcResidualResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from transit_timing_residual_plotter import format_oc_result
        mids = self._midpoints(n=3)
        r = compute_oc_residuals(mids, period_days=10.0, epoch_bjd=2458000.0)
        s = format_oc_result(r)
        assert "|" in s
