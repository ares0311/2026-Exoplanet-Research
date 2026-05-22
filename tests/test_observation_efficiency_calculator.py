"""Tests for Skills/observation_efficiency_calculator.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_efficiency_calculator import (
    ObsEfficiencyResult,
    compute_obs_efficiency,
    format_efficiency_result,
)


def _uniform_times(n: int = 100, dt: float = 0.5) -> list[float]:
    return [i * dt for i in range(n)]


class TestComputeObsEfficiency:
    def test_returns_obs_efficiency_result(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        assert isinstance(result, ObsEfficiencyResult)

    def test_flag_ok_for_valid_input(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        assert result.flag == "OK"

    def test_too_few_points_returns_invalid(self):
        result = compute_obs_efficiency([0.0], 5.0, 0.0)
        assert result.flag == "INVALID"

    def test_empty_input_returns_invalid(self):
        result = compute_obs_efficiency([], 5.0, 0.0)
        assert result.flag == "INVALID"

    def test_negative_period_returns_invalid(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, -1.0, 0.0)
        assert result.flag == "INVALID"

    def test_coverage_fraction_between_zero_and_one(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        assert 0.0 <= result.coverage_fraction <= 1.0

    def test_max_gap_phase_between_zero_and_one(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        assert 0.0 <= result.max_gap_phase <= 1.0

    def test_uniform_coverage_high_fraction(self):
        # Many uniformly-spaced points should give high coverage
        time = [i * 0.05 for i in range(500)]
        result = compute_obs_efficiency(time, 5.0, 0.0, n_bins=20)
        assert result.coverage_fraction >= 0.8

    def test_is_sufficient_true_when_coverage_above_min(self):
        time = [i * 0.05 for i in range(500)]
        result = compute_obs_efficiency(time, 5.0, 0.0, min_coverage=0.5)
        assert result.is_sufficient

    def test_is_sufficient_false_when_sparse(self):
        # Only a few points
        time = [0.0, 5.0]
        result = compute_obs_efficiency(time, 5.0, 0.0, n_bins=20)
        assert not result.is_sufficient or result.flag in ("INSUFFICIENT", "INVALID")

    def test_period_stored(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 7.3, 0.0)
        assert result.period_days == pytest.approx(7.3)

    def test_n_phase_bins_stored(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0, n_bins=30)
        assert result.n_phase_bins == 30

    def test_frozen_dataclass(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatEfficiencyResult:
    def test_returns_string(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        md = format_efficiency_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        time = _uniform_times()
        result = compute_obs_efficiency(time, 5.0, 0.0)
        md = format_efficiency_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = compute_obs_efficiency([], 5.0, 0.0)
        md = format_efficiency_result(result)
        assert "INVALID" in md
