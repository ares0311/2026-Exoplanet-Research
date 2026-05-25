"""Tests for Skills/disposition_recorder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from disposition_recorder import (
    VALID_STATUSES,
    format_disposition_result,
    get_disposition_history,
    record_disposition,
)


class TestValidStatuses:
    def test_pc_in_valid(self):
        assert "PC" in VALID_STATUSES

    def test_fp_in_valid(self):
        assert "FP" in VALID_STATUSES

    def test_cp_in_valid(self):
        assert "CP" in VALID_STATUSES

    def test_eb_in_valid(self):
        assert "EB" in VALID_STATUSES

    def test_is_in_valid(self):
        assert "IS" in VALID_STATUSES

    def test_unk_in_valid(self):
        assert "UNK" in VALID_STATUSES


class TestRecordDisposition:
    def test_basic_record(self, tmp_path):
        store = tmp_path / "disp.json"
        result = record_disposition(12345, "PC", confidence=0.8, author="alice", store_path=store)
        assert result.flag == "OK"

    def test_invalid_status(self, tmp_path):
        store = tmp_path / "disp.json"
        result = record_disposition(12345, "INVALID_STATUS", author="alice", store_path=store)
        assert result.flag == "INVALID"

    def test_confidence_out_of_range_high(self, tmp_path):
        store = tmp_path / "disp.json"
        result = record_disposition(12345, "PC", confidence=1.5, author="alice", store_path=store)
        assert result.flag == "INVALID"

    def test_confidence_out_of_range_low(self, tmp_path):
        store = tmp_path / "disp.json"
        result = record_disposition(12345, "PC", confidence=-0.1, author="alice", store_path=store)
        assert result.flag == "INVALID"

    def test_confidence_stored(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "FP", confidence=0.9, author="bob", store_path=store)
        result = get_disposition_history(12345, store)
        assert result.current_confidence == 0.9

    def test_multiple_dispositions(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "PC", confidence=0.7, author="a", store_path=store)
        record_disposition(12345, "FP", confidence=0.9, author="b", store_path=store)
        result = get_disposition_history(12345, store)
        assert result.n_records == 2

    def test_latest_status_is_most_recent(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "PC", author="a", store_path=store)
        record_disposition(12345, "FP", author="b", store_path=store)
        result = get_disposition_history(12345, store)
        assert result.current_status == "FP"

    def test_note_stored(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "EB", note="odd-even asymmetry", author="c", store_path=store)
        result = get_disposition_history(12345, store)
        assert "odd-even" in result.history[0].note


class TestGetDispositionHistory:
    def test_no_history_returns_empty(self, tmp_path):
        store = tmp_path / "disp.json"
        result = get_disposition_history(99999, store)
        assert result.flag == "EMPTY"

    def test_tic_id_stored(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(42, "UNK", author="x", store_path=store)
        result = get_disposition_history(42, store)
        assert result.tic_id == 42

    def test_result_frozen(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(42, "PC", author="x", store_path=store)
        result = get_disposition_history(42, store)
        try:
            result.n_records = 99
            raise AssertionError()
        except Exception:
            pass


class TestFormatDispositionResult:
    def test_returns_string(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(42, "PC", author="x", store_path=store)
        result = get_disposition_history(42, store)
        text = format_disposition_result(result)
        assert isinstance(text, str)

    def test_contains_tic_id(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "CP", author="x", store_path=store)
        result = get_disposition_history(12345, store)
        text = format_disposition_result(result)
        assert "12345" in text

    def test_contains_status(self, tmp_path):
        store = tmp_path / "disp.json"
        record_disposition(12345, "CP", author="x", store_path=store)
        result = get_disposition_history(12345, store)
        text = format_disposition_result(result)
        assert "CP" in text
