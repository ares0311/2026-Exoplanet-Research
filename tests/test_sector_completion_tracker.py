"""Tests for Skills/sector_completion_tracker.py."""
import json
import pytest
from pathlib import Path
from Skills.sector_completion_tracker import (
    SectorCompletionLog,
    PIPELINE_STAGES,
    format_completion_report,
)


class TestSectorCompletionLog:
    def test_create_new_log(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        assert log.completion_summary()["total_entries"] == 0

    def test_mark_complete(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        assert log.is_complete(12345, 1, "fetch")

    def test_is_complete_false_before_mark(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        assert not log.is_complete(12345, 1, "fetch")

    def test_mark_multiple_stages(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        for stage in ("fetch", "clean", "search"):
            log.mark_complete(12345, 1, stage)
        stages = log.completed_stages(12345, 1)
        assert stages == {"fetch", "clean", "search"}

    def test_mark_idempotent(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        log.mark_complete(12345, 1, "fetch")
        assert len(log.completed_stages(12345, 1)) == 1

    def test_persistence_across_instances(self, tmp_path):
        p = tmp_path / "log.json"
        log1 = SectorCompletionLog(p)
        log1.mark_complete(12345, 1, "fetch")
        log2 = SectorCompletionLog(p)
        assert log2.is_complete(12345, 1, "fetch")

    def test_completion_summary_counts(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        for stage in PIPELINE_STAGES:
            log.mark_complete(12345, 1, stage)
        summary = log.completion_summary()
        assert summary["fully_complete"] == 1
        assert summary["total_entries"] == 1

    def test_completion_summary_stage_counts(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        log.mark_complete(99999, 2, "fetch")
        summary = log.completion_summary()
        assert summary["stage_counts"]["fetch"] == 2

    def test_export_incomplete(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        incomplete = log.export_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0]["tic_id"] == 12345
        assert "fetch" not in incomplete[0]["missing"]
        assert "clean" in incomplete[0]["missing"]

    def test_export_incomplete_fully_done_excluded(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        for stage in PIPELINE_STAGES:
            log.mark_complete(12345, 1, stage)
        incomplete = log.export_incomplete()
        assert len(incomplete) == 0

    def test_multiple_sectors_same_tic(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        log.mark_complete(12345, 2, "fetch")
        summary = log.completion_summary()
        assert summary["total_entries"] == 2

    def test_json_file_created(self, tmp_path):
        p = tmp_path / "subdir" / "log.json"
        log = SectorCompletionLog(p)
        log.mark_complete(1, 1, "fetch")
        assert p.exists()

    def test_export_incomplete_custom_stages(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(12345, 1, "fetch")
        incomplete = log.export_incomplete(required_stages=["fetch", "clean"])
        assert incomplete[0]["missing"] == ["clean"]


class TestFormatCompletionReport:
    def test_returns_string(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        s = format_completion_report(log)
        assert isinstance(s, str)

    def test_contains_header(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        s = format_completion_report(log)
        assert "Sector" in s and "Completion" in s

    def test_contains_stage_names(self, tmp_path):
        log = SectorCompletionLog(tmp_path / "log.json")
        log.mark_complete(1, 1, "fetch")
        s = format_completion_report(log)
        assert "fetch" in s
