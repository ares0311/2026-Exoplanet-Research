"""Tests for candidate_flag_summary.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_flag_summary import (
    FAIL,
    PASS,
    WARN,
    FlagEntry,
    format_flag_summary,
    summarise_flags,
)


class TestSummariseFlags:
    def test_empty_returns_empty_flag(self):
        r = summarise_flags([])
        assert r.flag == "EMPTY"
        assert r.verdict == PASS

    def test_all_pass(self):
        flags = [FlagEntry("centroid", PASS), FlagEntry("depth", PASS)]
        r = summarise_flags(flags)
        assert r.verdict == PASS
        assert r.n_pass == 2
        assert r.n_warn == 0
        assert r.n_fail == 0

    def test_warn_escalates_verdict(self):
        flags = [FlagEntry("centroid", PASS), FlagEntry("odd_even", WARN)]
        r = summarise_flags(flags)
        assert r.verdict == WARN

    def test_fail_escalates_verdict(self):
        flags = [FlagEntry("centroid", PASS), FlagEntry("sec_eclipse", FAIL)]
        r = summarise_flags(flags)
        assert r.verdict == FAIL

    def test_fail_dominates_warn(self):
        flags = [
            FlagEntry("a", WARN),
            FlagEntry("b", FAIL),
            FlagEntry("c", WARN),
        ]
        r = summarise_flags(flags)
        assert r.verdict == FAIL

    def test_counts_correct(self):
        flags = [FlagEntry("a", PASS), FlagEntry("b", WARN), FlagEntry("c", FAIL)]
        r = summarise_flags(flags)
        assert r.n_pass == 1
        assert r.n_warn == 1
        assert r.n_fail == 1

    def test_invalid_level_treated_as_warn(self):
        flags = [FlagEntry("x", "UNKNOWN")]
        r = summarise_flags(flags)
        assert r.n_warn == 1
        assert r.flags[0].level == WARN

    def test_flags_preserved(self):
        flags = [
            FlagEntry("depth", PASS, "depth ok"),
            FlagEntry("centroid", FAIL, "shift detected"),
        ]
        r = summarise_flags(flags)
        assert len(r.flags) == 2

    def test_result_frozen(self):
        r = summarise_flags([FlagEntry("x", PASS)])
        try:
            r.verdict = "FAIL"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_ok_status_on_non_empty(self):
        r = summarise_flags([FlagEntry("x", PASS)])
        assert r.flag == "OK"


class TestFormatFlagSummary:
    def test_returns_string(self):
        r = summarise_flags([FlagEntry("depth", PASS)])
        assert isinstance(format_flag_summary(r), str)

    def test_contains_verdict(self):
        r = summarise_flags([FlagEntry("depth", FAIL)])
        assert "FAIL" in format_flag_summary(r)
