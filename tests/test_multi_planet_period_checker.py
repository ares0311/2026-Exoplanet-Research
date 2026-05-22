"""Tests for multi_planet_period_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from multi_planet_period_checker import (
    check_multi_planet_periods,
    format_multi_planet_check,
)


class TestCheckMultiPlanetPeriods:
    def test_result_frozen(self):
        r = check_multi_planet_periods([3.0, 7.0])
        try:
            r.n_candidates = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_pair_result_frozen(self):
        r = check_multi_planet_periods([3.0, 6.0])
        if r.pairs:
            pr = r.pairs[0]
            try:
                pr.is_harmonic = False  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_empty_periods_invalid(self):
        r = check_multi_planet_periods([])
        assert r.flag == "INVALID"

    def test_negative_period_invalid(self):
        r = check_multi_planet_periods([-1.0])
        assert r.flag == "INVALID"

    def test_single_period_single_flag(self):
        r = check_multi_planet_periods([5.0])
        assert r.flag == "SINGLE"
        assert r.n_candidates == 1

    def test_two_independent_periods(self):
        r = check_multi_planet_periods([3.0, 7.3])
        assert r.n_harmonic_pairs == 0
        assert r.flag == "OK"

    def test_2_to_1_harmonic(self):
        r = check_multi_planet_periods([5.0, 10.0])
        assert any(pr.is_harmonic for pr in r.pairs)
        assert r.n_harmonic_pairs >= 1

    def test_3_to_1_harmonic(self):
        r = check_multi_planet_periods([5.0, 15.0])
        assert any(pr.is_harmonic for pr in r.pairs)

    def test_alias_same_period(self):
        r = check_multi_planet_periods([5.0, 5.01])
        assert any(pr.relationship == "alias" for pr in r.pairs)

    def test_n_pairs_correct(self):
        r = check_multi_planet_periods([3.0, 5.0, 7.0])
        assert len(r.pairs) <= 3

    def test_n_candidates_correct(self):
        r = check_multi_planet_periods([3.0, 5.0, 7.0])
        assert r.n_candidates == 3

    def test_ratio_always_ge_1(self):
        r = check_multi_planet_periods([3.0, 7.0, 12.0])
        for pr in r.pairs:
            assert pr.ratio >= 1.0

    def test_format_returns_string(self):
        r = check_multi_planet_periods([3.0, 6.0])
        s = format_multi_planet_check(r)
        assert isinstance(s, str)
        assert "Period" in s

    def test_format_contains_table_when_pairs(self):
        r = check_multi_planet_periods([3.0, 6.0])
        s = format_multi_planet_check(r)
        if r.pairs:
            assert "|" in s
