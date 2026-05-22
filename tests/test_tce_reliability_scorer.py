"""Tests for Skills/tce_reliability_scorer.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tce_reliability_scorer import (
    TCEReliabilityResult,
    format_reliability_result,
    score_tce_reliability,
)


class TestScoreTceReliability:
    def test_returns_tce_reliability_result(self):
        result = score_tce_reliability(10.0, 5)
        assert isinstance(result, TCEReliabilityResult)

    def test_flag_ok_for_valid_input(self):
        result = score_tce_reliability(10.0, 5)
        assert result.flag == "OK"

    def test_negative_mes_returns_invalid(self):
        result = score_tce_reliability(-1.0, 5)
        assert result.flag == "INVALID"

    def test_no_inputs_returns_insufficient(self):
        result = score_tce_reliability(None, None)
        assert result.flag == "INSUFFICIENT"

    def test_composite_score_in_zero_one_range(self):
        result = score_tce_reliability(12.0, 8)
        assert 0.0 <= result.composite_score <= 1.0

    def test_high_mes_high_transits_is_reliable(self):
        result = score_tce_reliability(15.0, 10)
        assert result.is_reliable

    def test_below_mes_threshold_caps_score(self):
        result = score_tce_reliability(5.0, 10, mes_threshold=7.1)
        assert result.composite_score <= 0.4

    def test_below_n_transit_min_caps_score(self):
        result = score_tce_reliability(15.0, 2, n_transit_min=3)
        assert result.composite_score <= 0.3

    def test_ses_mes_ratio_affects_score(self):
        result_good = score_tce_reliability(12.0, 8, ses_mes_ratio=0.3)
        result_bad = score_tce_reliability(12.0, 8, ses_mes_ratio=0.99)
        assert result_good.composite_score >= result_bad.composite_score

    def test_fit_residual_affects_score(self):
        result_good = score_tce_reliability(12.0, 8, fit_residual_ppm=50.0)
        result_bad = score_tce_reliability(12.0, 8, fit_residual_ppm=5000.0)
        assert result_good.composite_score >= result_bad.composite_score

    def test_mes_stored(self):
        result = score_tce_reliability(10.5, 5)
        assert result.mes == pytest.approx(10.5)

    def test_n_transits_stored(self):
        result = score_tce_reliability(10.0, 7)
        assert result.n_transits == 7

    def test_reliability_threshold_respected(self):
        result = score_tce_reliability(8.0, 4, reliability_threshold=0.6)
        assert result.is_reliable == (result.composite_score >= 0.6)

    def test_frozen_dataclass(self):
        result = score_tce_reliability(10.0, 5)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatReliabilityResult:
    def test_returns_string(self):
        result = score_tce_reliability(10.0, 5)
        md = format_reliability_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = score_tce_reliability(10.0, 5)
        md = format_reliability_result(result)
        assert result.flag in md

    def test_insufficient_format(self):
        result = score_tce_reliability(None, None)
        md = format_reliability_result(result)
        assert "INSUFFICIENT" in md
