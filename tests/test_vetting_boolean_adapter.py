"""Tests for Skills/vetting_boolean_adapter.py."""
import pytest
from Skills.vetting_boolean_adapter import (
    FlagEntry,
    VettingTriageResult,
    boolean_flags_to_entries,
    run_vetting_triage,
    format_triage_result,
)


class TestBooleanFlagsToEntries:
    def test_returns_list(self):
        entries = boolean_flags_to_entries({"centroid_shift": True})
        assert isinstance(entries, list)

    def test_entry_type(self):
        entries = boolean_flags_to_entries({"centroid_shift": True})
        assert isinstance(entries[0], FlagEntry)

    def test_known_flag_gets_category(self):
        entries = boolean_flags_to_entries({"centroid_shift": True})
        assert entries[0].category == "centroid"
        assert entries[0].severity == "critical"

    def test_unknown_flag_gets_info(self):
        entries = boolean_flags_to_entries({"unknown_flag": True})
        assert entries[0].category == "unknown"
        assert entries[0].severity == "info"

    def test_value_preserved(self):
        entries = boolean_flags_to_entries({"centroid_shift": False})
        assert entries[0].value is False

    def test_multiple_flags(self):
        flags = {"centroid_shift": True, "stellar_variability": False}
        entries = boolean_flags_to_entries(flags)
        assert len(entries) == 2

    def test_custom_severity_map(self):
        custom = {"my_flag": ("photometry", "warning")}
        entries = boolean_flags_to_entries({"my_flag": True}, severity_map=custom)
        assert entries[0].severity == "warning"
        assert entries[0].category == "photometry"

    def test_empty_dict_returns_empty(self):
        entries = boolean_flags_to_entries({})
        assert entries == []


class TestRunVettingTriage:
    def test_returns_result_type(self):
        result = run_vetting_triage({"centroid_shift": True})
        assert isinstance(result, VettingTriageResult)

    def test_flag_ok(self):
        result = run_vetting_triage({"centroid_shift": False})
        assert result.flag == "OK"

    def test_flag_empty_for_empty_dict(self):
        result = run_vetting_triage({})
        assert result.flag == "EMPTY"
        assert result.triage_decision == "PASS"

    def test_flag_invalid_for_non_dict(self):
        result = run_vetting_triage("bad")
        assert result.flag == "INVALID"
        assert result.triage_decision == "FAIL"

    def test_critical_flag_raised_fail(self):
        result = run_vetting_triage({"centroid_shift": True})
        assert result.triage_decision == "FAIL"
        assert result.n_critical == 1

    def test_warning_only_warn(self):
        result = run_vetting_triage({"stellar_variability": True})
        assert result.triage_decision == "WARN"
        assert result.n_warning == 1

    def test_all_false_pass(self):
        result = run_vetting_triage({"centroid_shift": False, "stellar_variability": False})
        assert result.triage_decision == "PASS"
        assert result.n_raised == 0

    def test_n_flags_count(self):
        result = run_vetting_triage({"centroid_shift": True, "stellar_variability": False})
        assert result.n_flags == 2

    def test_n_raised_count(self):
        result = run_vetting_triage({"centroid_shift": True, "stellar_variability": True})
        assert result.n_raised == 2

    def test_raised_flags_tuple(self):
        result = run_vetting_triage({"centroid_shift": True})
        assert len(result.raised_flags) == 1
        assert result.raised_flags[0].name == "centroid_shift"

    def test_critical_overrides_warning(self):
        result = run_vetting_triage({"centroid_shift": True, "stellar_variability": True})
        assert result.triage_decision == "FAIL"

    def test_info_flag_raised_still_pass(self):
        result = run_vetting_triage({"period_alias_possible": True})
        assert result.triage_decision == "PASS"
        assert result.n_info == 1


class TestFormatTriageResult:
    def test_returns_string(self):
        result = run_vetting_triage({"centroid_shift": True})
        s = format_triage_result(result)
        assert isinstance(s, str)

    def test_contains_decision(self):
        result = run_vetting_triage({"centroid_shift": True})
        s = format_triage_result(result)
        assert "FAIL" in s

    def test_contains_raised_flags(self):
        result = run_vetting_triage({"centroid_shift": True})
        s = format_triage_result(result)
        assert "centroid_shift" in s
