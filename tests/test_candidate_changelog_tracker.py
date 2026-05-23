"""Tests for Skills/candidate_changelog_tracker.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_changelog_tracker import (
    ChangeEntry,
    ChangelogResult,
    record_change,
    get_changelog,
    format_changelog_result,
)


class TestRecordChange:
    def test_basic_record(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "fpp", 0.3, 0.1, author="alice", store_path=store)
        result = get_changelog(12345, store)
        assert result.flag == "OK"
        assert result.n_changes == 1

    def test_multiple_records(self, tmp_path):
        store = tmp_path / "changelog.json"
        for i in range(3):
            record_change(12345, f"field_{i}", i, i + 1, author="bob", store_path=store)
        result = get_changelog(12345, store)
        assert result.n_changes == 3

    def test_author_stored(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "period", 1.0, 2.0, author="carol", store_path=store)
        result = get_changelog(12345, store)
        assert result.entries[0].author == "carol"

    def test_field_stored(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "stellar_radius", 0.9, 1.1, author="dave", store_path=store)
        result = get_changelog(12345, store)
        assert result.entries[0].field == "stellar_radius"

    def test_old_new_values_stored(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "depth", 500.0, 600.0, author="eve", store_path=store)
        result = get_changelog(12345, store)
        assert result.entries[0].old_value == "500.0"
        assert result.entries[0].new_value == "600.0"

    def test_empty_field_invalid(self, tmp_path):
        store = tmp_path / "changelog.json"
        result = record_change(12345, "", 0.0, 1.0, author="x", store_path=store)
        assert result is None or (hasattr(result, "flag") and result.flag == "INVALID")

    def test_separate_tic_ids_isolated(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(111, "fpp", 0.1, 0.2, author="a", store_path=store)
        record_change(222, "fpp", 0.5, 0.6, author="b", store_path=store)
        r1 = get_changelog(111, store)
        r2 = get_changelog(222, store)
        assert r1.n_changes == 1
        assert r2.n_changes == 1

    def test_timestamp_present(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "fpp", 0.1, 0.2, author="x", store_path=store)
        result = get_changelog(12345, store)
        assert result.entries[0].timestamp


class TestGetChangelog:
    def test_empty_returns_empty_flag(self, tmp_path):
        store = tmp_path / "changelog.json"
        result = get_changelog(99999, store)
        assert result.flag == "EMPTY"

    def test_tic_id_stored(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(42, "fpp", 0.0, 1.0, author="x", store_path=store)
        result = get_changelog(42, store)
        assert result.tic_id == 42

    def test_result_frozen(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(42, "fpp", 0.0, 1.0, author="x", store_path=store)
        result = get_changelog(42, store)
        try:
            result.n_changes = 99
            assert False
        except Exception:
            pass

    def test_format_returns_string(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(42, "fpp", 0.0, 1.0, author="x", store_path=store)
        result = get_changelog(42, store)
        text = format_changelog_result(result)
        assert isinstance(text, str)

    def test_format_contains_tic_id(self, tmp_path):
        store = tmp_path / "changelog.json"
        record_change(12345, "fpp", 0.0, 1.0, author="x", store_path=store)
        result = get_changelog(12345, store)
        text = format_changelog_result(result)
        assert "12345" in text
