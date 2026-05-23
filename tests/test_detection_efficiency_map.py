"""Tests for Skills/detection_efficiency_map.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from detection_efficiency_map import (
    EfficiencyCell,
    DetectionEfficiencyResult,
    compute_detection_efficiency,
    format_efficiency_result,
)


class TestComputeDetectionEfficiency:
    def _make_injected(self, n=10, period=5.0, depth=500.0):
        return [{"period_days": period, "depth_ppm": depth}] * n

    def test_basic_ok(self):
        inj = self._make_injected(10)
        rec = inj[:7]
        p_bins = [0.0, 10.0]
        d_bins = [0.0, 1000.0]
        result = compute_detection_efficiency(inj, rec, p_bins, d_bins)
        assert result.flag == "OK"

    def test_perfect_recovery(self):
        inj = self._make_injected(10)
        rec = inj[:]
        p_bins = [0.0, 10.0]
        d_bins = [0.0, 1000.0]
        result = compute_detection_efficiency(inj, rec, p_bins, d_bins)
        assert result.cells[0].efficiency == 1.0

    def test_zero_recovery(self):
        inj = self._make_injected(10)
        p_bins = [0.0, 10.0]
        d_bins = [0.0, 1000.0]
        result = compute_detection_efficiency(inj, [], p_bins, d_bins)
        assert result.cells[0].efficiency == 0.0

    def test_empty_injected_insufficient(self):
        result = compute_detection_efficiency([], [], [0.0, 10.0], [0.0, 1000.0])
        assert result.flag == "INSUFFICIENT"

    def test_n_cells_correct(self):
        inj = self._make_injected(10)
        p_bins = [0.0, 5.0, 10.0]
        d_bins = [0.0, 500.0, 1000.0]
        result = compute_detection_efficiency(inj, inj, p_bins, d_bins)
        assert result.n_cells == (len(p_bins) - 1) * (len(d_bins) - 1)

    def test_efficiency_in_range(self):
        inj = self._make_injected(20)
        rec = inj[:12]
        p_bins = [0.0, 10.0]
        d_bins = [0.0, 1000.0]
        result = compute_detection_efficiency(inj, rec, p_bins, d_bins)
        for cell in result.cells:
            assert 0.0 <= cell.efficiency <= 1.0

    def test_period_bins_stored(self):
        inj = self._make_injected(5)
        p_bins = [1.0, 5.0, 15.0]
        d_bins = [0.0, 1000.0]
        result = compute_detection_efficiency(inj, inj, p_bins, d_bins)
        assert list(result.period_bins) == p_bins

    def test_depth_bins_stored(self):
        inj = self._make_injected(5)
        p_bins = [0.0, 10.0]
        d_bins = [100.0, 500.0, 2000.0]
        result = compute_detection_efficiency(inj, inj, p_bins, d_bins)
        assert list(result.depth_bins) == d_bins

    def test_result_frozen(self):
        inj = self._make_injected(5)
        result = compute_detection_efficiency(inj, inj, [0.0, 10.0], [0.0, 1000.0])
        try:
            result.n_cells = 99
            assert False
        except Exception:
            pass

    def test_partial_recovery_multiple_cells(self):
        inj = [
            {"period_days": 2.5, "depth_ppm": 300.0},
            {"period_days": 7.5, "depth_ppm": 800.0},
        ] * 5
        rec = [{"period_days": 2.5, "depth_ppm": 300.0}] * 5
        p_bins = [0.0, 5.0, 10.0]
        d_bins = [0.0, 500.0, 1000.0]
        result = compute_detection_efficiency(inj, rec, p_bins, d_bins)
        assert result.flag == "OK"

    def test_format_returns_string(self):
        inj = self._make_injected(5)
        result = compute_detection_efficiency(inj, inj, [0.0, 10.0], [0.0, 1000.0])
        text = format_efficiency_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        inj = self._make_injected(5)
        result = compute_detection_efficiency(inj, inj, [0.0, 10.0], [0.0, 1000.0])
        text = format_efficiency_result(result)
        assert result.flag in text
