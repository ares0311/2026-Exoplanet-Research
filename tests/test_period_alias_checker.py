"""Tests for Skills.period_alias_checker."""
from __future__ import annotations

import pytest
from Skills.period_alias_checker import AliasCheckResult, check_period_alias, format_alias_result


class TestCheckPeriodAlias:
    def test_tess_half_orbit_alias(self) -> None:
        result = check_period_alias(13.7)
        assert result.is_alias
        assert result.alias_name == "TESS_half_orbit"

    def test_tess_full_orbit_alias(self) -> None:
        result = check_period_alias(27.4)
        assert result.is_alias

    def test_double_tess_orbit_is_harmonic(self) -> None:
        # 2 × 13.7 = 27.4 d, ratio = 27.4/13.7 = 2 (integer harmonic)
        result = check_period_alias(27.4, tol=0.05)
        assert result.is_alias

    def test_clean_period_not_flagged(self) -> None:
        result = check_period_alias(10.0)
        assert not result.is_alias

    def test_confidence_near_one_for_exact_match(self) -> None:
        result = check_period_alias(13.7)
        assert result.confidence > 0.9

    def test_confidence_lower_for_near_match(self) -> None:
        result = check_period_alias(13.3, tol=0.05)
        if result.is_alias:
            assert result.confidence < 0.9

    def test_returns_alias_check_result(self) -> None:
        result = check_period_alias(5.0)
        assert isinstance(result, AliasCheckResult)

    def test_alias_period_set_when_alias(self) -> None:
        result = check_period_alias(27.4)
        assert result.alias_period is not None

    def test_alias_period_none_when_clean(self) -> None:
        result = check_period_alias(10.0)
        assert result.alias_period is None

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError):
            check_period_alias(-1.0)

    def test_custom_alias_table(self) -> None:
        custom = [("my_alias", 7.5)]
        result = check_period_alias(7.5, known_aliases=custom)
        assert result.is_alias
        assert result.alias_name == "my_alias"

    def test_half_period_sub_harmonic(self) -> None:
        # 13.7 / 2 = 6.85; ratio = 6.85/13.7 = 0.5 = 1/2
        result = check_period_alias(6.85, tol=0.05)
        assert result.is_alias


class TestFormatAliasResult:
    def test_clean_message(self) -> None:
        result = check_period_alias(10.0)
        text = format_alias_result(result)
        assert "clean" in text.lower() or "no known" in text.lower()

    def test_alias_message_contains_name(self) -> None:
        result = check_period_alias(13.7)
        text = format_alias_result(result)
        assert "TESS_half_orbit" in text
