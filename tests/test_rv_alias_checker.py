"""Tests for Skills/rv_alias_checker.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_alias_checker import (
    RVAliasResult,
    check_rv_aliases,
    format_rv_alias_result,
)


class TestCheckRVAliases:
    def test_returns_result_type(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert isinstance(r, RVAliasResult)

    def test_flag_ok(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert r.flag == "OK"

    def test_candidate_period_stored(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert r.candidate_period_days == pytest.approx(3.5)

    def test_one_day_alias_detected(self):
        # A period very close to 1 day should be flagged
        r = check_rv_aliases(candidate_period_days=1.001)
        assert r.is_likely_alias

    def test_annual_alias_detected(self):
        # Very close to 365.25 days
        r = check_rv_aliases(candidate_period_days=365.24)
        assert r.is_likely_alias

    def test_unusual_period_not_alias(self):
        # A period far from known aliases
        r = check_rv_aliases(candidate_period_days=123.456)
        assert not r.is_likely_alias

    def test_alias_periods_tuple(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert isinstance(r.alias_periods_days, tuple)

    def test_alias_labels_tuple(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert isinstance(r.alias_labels, tuple)

    def test_alias_periods_and_labels_same_length(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert len(r.alias_periods_days) == len(r.alias_labels)

    def test_closest_alias_label_string(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert isinstance(r.closest_alias_label, str)

    def test_proximity_percent_finite(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        assert math.isfinite(r.alias_proximity_percent)
        assert r.alias_proximity_percent >= 0.0

    def test_custom_threshold_tighter(self):
        r_loose = check_rv_aliases(candidate_period_days=1.05, proximity_threshold_percent=10.0)
        r_tight = check_rv_aliases(candidate_period_days=1.05, proximity_threshold_percent=1.0)
        assert r_loose.is_likely_alias
        assert not r_tight.is_likely_alias

    def test_invalid_period(self):
        r = check_rv_aliases(candidate_period_days=0.0)
        assert r.flag != "OK"
        assert math.isnan(r.closest_alias_days)

    def test_user_supplied_aliases(self):
        r = check_rv_aliases(candidate_period_days=30.0, sampling_aliases=[30.1])
        assert r.flag == "OK"

    def test_frozen_dataclass(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        try:
            r.is_likely_alias = False  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatRVAliasResult:
    def test_ok_returns_table(self):
        r = check_rv_aliases(candidate_period_days=3.5)
        out = format_rv_alias_result(r)
        assert "Candidate period" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = check_rv_aliases(candidate_period_days=0.0)
        out = format_rv_alias_result(r)
        assert "flag=" in out
