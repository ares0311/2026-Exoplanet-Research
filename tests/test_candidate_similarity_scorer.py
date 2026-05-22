"""Tests for Skills/candidate_similarity_scorer.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_similarity_scorer import (
    SimilarityResult,
    format_similarity_result,
    score_similarity,
)


class TestScoreSimilarity:
    def test_returns_similarity_result(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert isinstance(result, SimilarityResult)

    def test_identical_periods_duplicate(self):
        a = {"period_days": 5.0, "depth_ppm": 1000.0, "duration_hours": 2.0}
        b = {"period_days": 5.0, "depth_ppm": 1000.0, "duration_hours": 2.0}
        result = score_similarity(a, b)
        assert result.is_duplicate
        assert result.relationship == "duplicate"

    def test_very_different_periods_unrelated(self):
        a = {"period_days": 5.0}
        b = {"period_days": 50.0}
        result = score_similarity(a, b)
        assert not result.is_duplicate
        assert result.relationship == "unrelated"

    def test_missing_period_returns_invalid(self):
        a = {"depth_ppm": 1000.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert result.flag == "INVALID"

    def test_zero_period_returns_invalid(self):
        a = {"period_days": 0.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert result.flag == "INVALID"

    def test_period_similarity_close_to_one_for_identical(self):
        a = {"period_days": 10.0}
        b = {"period_days": 10.0}
        result = score_similarity(a, b)
        assert result.period_similarity == pytest.approx(1.0, abs=1e-4)

    def test_depth_similarity_none_when_missing(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert result.depth_similarity is None

    def test_duration_similarity_none_when_missing(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert result.duration_similarity is None

    def test_period_2to1_ratio_is_alias(self):
        a = {"period_days": 10.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b, period_tol_frac=0.02)
        assert result.relationship in ("alias", "unrelated")

    def test_composite_score_in_zero_one_range(self):
        a = {"period_days": 5.0, "depth_ppm": 1000.0}
        b = {"period_days": 5.1, "depth_ppm": 1100.0}
        result = score_similarity(a, b)
        assert 0.0 <= result.composite_score <= 1.0

    def test_flag_ok_for_valid_input(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        assert result.flag == "OK"

    def test_frozen_dataclass(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatSimilarityResult:
    def test_returns_string(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        md = format_similarity_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        a = {"period_days": 5.0}
        b = {"period_days": 5.0}
        result = score_similarity(a, b)
        md = format_similarity_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = score_similarity({}, {})
        md = format_similarity_result(result)
        assert "INVALID" in md
