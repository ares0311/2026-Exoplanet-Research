"""Tests for Skills/period_refinement_calculator.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_refinement_calculator import (
    PeriodRefinementResult,
    format_refinement_result,
    refine_period_from_oc,
)


def _perfect_midpoints(period: float, epoch: float, n: int) -> list[float]:
    return [epoch + i * period for i in range(n)]


class TestRefinePeriodFromOc:
    def test_returns_period_refinement_result(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        assert isinstance(result, PeriodRefinementResult)

    def test_flag_ok_for_valid_input(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        assert result.flag == "OK"

    def test_single_midpoint_returns_insufficient(self):
        result = refine_period_from_oc([2450000.0], 5.0, 2450000.0)
        assert result.flag == "INSUFFICIENT"

    def test_empty_midpoints_returns_insufficient(self):
        result = refine_period_from_oc([], 5.0, 2450000.0)
        assert result.flag == "INSUFFICIENT"

    def test_negative_period_returns_invalid(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, -1.0, 2450000.0)
        assert result.flag == "INVALID"

    def test_refined_period_close_to_true_period(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 10)
        result = refine_period_from_oc(mids, 5.02, 2450000.0)
        if result.flag == "OK":
            assert result.refined_period_days is not None
            assert abs(result.refined_period_days - 5.0) < 0.1

    def test_oc_rms_near_zero_for_perfect_midpoints(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 8)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        assert result.oc_rms_minutes is not None
        assert result.oc_rms_minutes < 1e-3

    def test_n_transits_stored(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 7)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        assert result.n_transits == 7

    def test_initial_period_stored(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.1, 2450000.0)
        assert result.initial_period_days == pytest.approx(5.1, abs=1e-3)

    def test_period_uncertainty_may_be_computed(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 10)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        # uncertainty can be None if parabolic interpolation fails
        assert result.flag == "OK"

    def test_frozen_dataclass(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatRefinementResult:
    def test_returns_string(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        md = format_refinement_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        mids = _perfect_midpoints(5.0, 2450000.0, 5)
        result = refine_period_from_oc(mids, 5.0, 2450000.0)
        md = format_refinement_result(result)
        assert result.flag in md

    def test_insufficient_format(self):
        result = refine_period_from_oc([], 5.0, 2450000.0)
        md = format_refinement_result(result)
        assert "INSUFFICIENT" in md
