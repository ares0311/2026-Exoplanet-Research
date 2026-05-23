"""Tests for Skills/period_commensurability_checker.py."""
import pytest
from Skills.period_commensurability_checker import (
    CommensurabilityPair,
    CommensurabilityResult,
    check_commensurability,
    format_commensurability_result,
)


class TestCheckCommensurability:
    def test_returns_result_type(self):
        result = check_commensurability([3.0, 6.0])
        assert isinstance(result, CommensurabilityResult)

    def test_flag_ok(self):
        result = check_commensurability([3.0, 6.0])
        assert result.flag == "OK"

    def test_exact_2_1_resonance(self):
        result = check_commensurability([3.0, 6.0], tol_frac=0.01)
        assert result.n_resonant >= 1

    def test_exact_3_2_resonance(self):
        result = check_commensurability([4.0, 6.0], tol_frac=0.01)
        assert result.n_resonant >= 1

    def test_non_resonant(self):
        result = check_commensurability([3.0, 7.3], tol_frac=0.01)
        assert result.n_resonant == 0

    def test_single_period_no_pairs(self):
        result = check_commensurability([5.0])
        assert result.n_resonant == 0
        assert len(result.pairs) == 0

    def test_empty_list(self):
        result = check_commensurability([])
        assert result.flag == "OK"
        assert result.n_resonant == 0

    def test_pair_fields(self):
        result = check_commensurability([3.0, 6.0], tol_frac=0.05)
        if result.pairs:
            p = result.pairs[0]
            assert isinstance(p, CommensurabilityPair)
            assert p.flag == "OK"
            assert p.actual_ratio > 0
            assert p.deviation >= 0

    def test_custom_ratios(self):
        result = check_commensurability([5.0, 10.0], ratios=[(2, 1)], tol_frac=0.01)
        assert result.n_resonant >= 1

    def test_three_periods_multiple_pairs(self):
        result = check_commensurability([3.0, 6.0, 9.0], tol_frac=0.02)
        assert len(result.pairs) >= 2

    def test_n_periods(self):
        result = check_commensurability([3.0, 6.0, 9.0])
        assert result.n_periods == 3

    def test_near_resonance_flag_true(self):
        result = check_commensurability([3.0, 6.01], tol_frac=0.05)
        resonant = [p for p in result.pairs if p.is_near_resonance]
        assert len(resonant) >= 1

    def test_invalid_period_zero(self):
        result = check_commensurability([0.0, 3.0])
        assert result.flag in ("OK", "INVALID")


class TestFormatCommensurabilityResult:
    def test_returns_string(self):
        result = check_commensurability([3.0, 6.0])
        s = format_commensurability_result(result)
        assert isinstance(s, str)

    def test_contains_resonant_count(self):
        result = check_commensurability([3.0, 6.0])
        s = format_commensurability_result(result)
        assert str(result.n_resonant) in s
