"""Tests for Skills/period_alias_resolver.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_alias_resolver import (
    AliasResolution,
    resolve_period_alias,
    format_alias_resolution,
)


def test_consistent_periods():
    result = resolve_period_alias(5.0, 5.01, period_rtol=0.01)
    assert result.alias_type == "CONSISTENT"
    assert result.flag == "OK"


def test_harmonic_2x():
    result = resolve_period_alias(5.0, 10.0, period_rtol=0.01)
    assert result.alias_type == "HARMONIC"
    assert result.flag == "ALIAS_DETECTED"


def test_harmonic_3x():
    result = resolve_period_alias(5.0, 15.0, period_rtol=0.01)
    assert result.alias_type == "HARMONIC"


def test_unrelated_periods():
    result = resolve_period_alias(5.0, 7.3, period_rtol=0.01)
    assert result.alias_type == "UNRELATED"
    assert result.flag == "UNRESOLVED"


def test_ratio_computed():
    result = resolve_period_alias(5.0, 10.0, period_rtol=0.01)
    assert abs(result.ratio - 2.0) < 0.01


def test_resolved_period_is_longer():
    result = resolve_period_alias(5.0, 10.0, period_rtol=0.01)
    assert result.resolved_period >= 5.0


def test_invalid_periods():
    result = resolve_period_alias(0.0, 5.0)
    assert result.flag == "UNRESOLVED"


def test_negative_period():
    result = resolve_period_alias(-1.0, 5.0)
    assert result.flag == "UNRESOLVED"


def test_half_harmonic():
    # P_a = 10, P_b = 5 → ratio = 0.5 → nearest = 0.5 (half-integer check)
    result = resolve_period_alias(10.0, 5.0, period_rtol=0.01)
    # After sorting pa=5, pb=10; ratio=2 → harmonic
    assert result.flag in ("OK", "ALIAS_DETECTED")


def test_delta_frac_small_for_consistent():
    result = resolve_period_alias(5.0, 5.005, period_rtol=0.01)
    assert result.delta_frac < 0.01


def test_format_returns_string():
    result = resolve_period_alias(5.0, 10.0)
    text = format_alias_resolution(result)
    assert isinstance(text, str)
    assert "Alias" in text


def test_format_has_table():
    result = resolve_period_alias(5.0, 10.0)
    text = format_alias_resolution(result)
    assert "|" in text
