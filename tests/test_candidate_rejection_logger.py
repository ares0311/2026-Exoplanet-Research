"""Tests for Skills/candidate_rejection_logger.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_rejection_logger import RejectionLog, format_rejection_summary


class TestRejectionLog:
    def _tmp_log(self, tmp_path: str) -> RejectionLog:
        return RejectionLog(os.path.join(tmp_path, "rejections.json"))

    def test_record_and_get(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(123, "HIGH_FPP", fpp=0.95)
        entries = log.get(123)
        assert len(entries) == 1
        assert entries[0].tic_id == 123

    def test_reason_code_stored(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(456, "LOW_SNR")
        assert log.get(456)[0].reason_code == "LOW_SNR"

    def test_fpp_stored(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(789, "HIGH_FPP", fpp=0.80)
        assert log.get(789)[0].fpp == 0.80

    def test_note_stored(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(111, "REASON", note="test note")
        assert log.get(111)[0].note == "test note"

    def test_all_entries(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(1, "A")
        log.record(2, "B")
        assert len(log.all_entries()) == 2

    def test_summary_count(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(1, "HIGH_FPP")
        log.record(2, "HIGH_FPP")
        log.record(3, "LOW_SNR")
        s = log.summary()
        assert s["n_rejected"] == 3
        assert s["reason_counts"]["HIGH_FPP"] == 2

    def test_export_csv_header(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(1, "X")
        csv = log.export_csv()
        assert csv.startswith("tic_id,reason_code")

    def test_empty_log_summary(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        s = log.summary()
        assert s["n_rejected"] == 0

    def test_get_nonexistent_tic(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        assert log.get(9999) == []

    def test_rejected_at_set(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        e = log.record(1, "X")
        assert len(e.rejected_at) > 0

    def test_multiple_records_same_tic(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(1, "A")
        log.record(1, "B")
        assert len(log.get(1)) == 2

    def test_format_returns_string(self, tmp_path) -> None:
        log = RejectionLog(tmp_path / "rej.json")
        log.record(1, "HIGH_FPP")
        s = format_rejection_summary(log)
        assert isinstance(s, str)
        assert "Rejection" in s
