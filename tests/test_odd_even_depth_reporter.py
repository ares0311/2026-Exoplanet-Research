"""Tests for Skills/odd_even_depth_reporter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from odd_even_depth_reporter import (
    compute_odd_even_significance,
    format_odd_even_result,
)


class TestOddEvenDepthReporter:
    """Tests for compute_odd_even_significance and format_odd_even_result."""

    def test_identical_depths_delta_zero(self) -> None:
        """Identical depths in both groups → delta = 0 and flag OK."""
        result = compute_odd_even_significance(
            [1000.0, 1000.0], [10.0, 10.0],
            [1000.0, 1000.0], [10.0, 10.0],
        )
        assert result.delta == 0.0
        assert result.flag == "OK"

    def test_significant_difference(self) -> None:
        """Large depth difference with small errors → SIGNIFICANT_ODD_EVEN."""
        result = compute_odd_even_significance(
            [1000.0], [1.0],
            [2000.0], [1.0],
        )
        assert result.flag == "SIGNIFICANT_ODD_EVEN"
        assert result.significance > 3.0

    def test_insufficient_odd_data(self) -> None:
        """Empty odd list → INSUFFICIENT_DATA."""
        result = compute_odd_even_significance(
            [], [],
            [1000.0], [10.0],
        )
        assert result.flag == "INSUFFICIENT_DATA"

    def test_insufficient_even_data(self) -> None:
        """Empty even list → INSUFFICIENT_DATA."""
        result = compute_odd_even_significance(
            [1000.0], [10.0],
            [], [],
        )
        assert result.flag == "INSUFFICIENT_DATA"

    def test_invalid_errors_zero(self) -> None:
        """Zero error value → INVALID_ERRORS."""
        result = compute_odd_even_significance(
            [1000.0], [0.0],
            [1000.0], [10.0],
        )
        assert result.flag == "INVALID_ERRORS"

    def test_significance_above_3_flag(self) -> None:
        """Significance > 3 is flagged as SIGNIFICANT_ODD_EVEN."""
        result = compute_odd_even_significance(
            [500.0], [1.0],
            [520.0], [1.0],
        )
        # delta = 20, sigma = sqrt(1+1) ≈ 1.414, significance ≈ 14.1
        assert result.flag == "SIGNIFICANT_ODD_EVEN"

    def test_flag_ok_for_similar_depths(self) -> None:
        """Similar depths with reasonable errors → OK."""
        result = compute_odd_even_significance(
            [1000.0, 1001.0], [50.0, 50.0],
            [999.0, 1002.0], [50.0, 50.0],
        )
        assert result.flag == "OK"

    def test_odd_mean_calculated_correctly(self) -> None:
        """Weighted mean for equal-weight single odd transit is the depth itself."""
        result = compute_odd_even_significance(
            [800.0], [10.0],
            [800.0], [10.0],
        )
        assert abs(result.odd_mean - 800.0) < 1e-9

    def test_sigma_greater_than_zero(self) -> None:
        """Combined sigma must be > 0 when errors are valid."""
        result = compute_odd_even_significance(
            [1000.0], [5.0],
            [1000.0], [5.0],
        )
        assert result.sigma > 0.0

    def test_delta_is_absolute_difference(self) -> None:
        """delta = |odd_mean - even_mean|, so always non-negative."""
        result = compute_odd_even_significance(
            [900.0], [10.0],
            [1100.0], [10.0],
        )
        assert result.delta >= 0.0
        assert abs(result.delta - abs(result.odd_mean - result.even_mean)) < 1e-9

    def test_format_returns_string(self) -> None:
        """format_odd_even_result returns a non-empty string."""
        result = compute_odd_even_significance(
            [1000.0], [10.0],
            [1000.0], [10.0],
        )
        text = format_odd_even_result(result)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_contains_significance(self) -> None:
        """Formatted output mentions 'Significance'."""
        result = compute_odd_even_significance(
            [1000.0], [10.0],
            [1050.0], [10.0],
        )
        text = format_odd_even_result(result)
        assert "Significance" in text
