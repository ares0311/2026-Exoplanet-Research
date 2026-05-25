"""Tests for Skills/training_curve_logger.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from training_curve_logger import (
    TrainingRunRecord,
    TrainingSummary,
    append_epoch,
    format_training_summary,
    load_training_log,
    summarize_training_log,
)

_TS = "2026-05-25T00:00:00Z"


def _make_record(run_id: str = "run1", epoch: int = 1, val_auc: float = 0.85) -> TrainingRunRecord:
    return TrainingRunRecord(
        run_id=run_id, epoch=epoch, train_loss=0.4, val_loss=0.5, val_auc=val_auc, timestamp=_TS
    )


def test_append_creates_file(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record())
    assert p.exists()


def test_second_append_does_not_overwrite(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record(epoch=1))
    append_epoch(p, _make_record(epoch=2))
    records = load_training_log(p)
    assert len(records) == 2


def test_load_returns_list(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record())
    records = load_training_log(p)
    assert isinstance(records, list)
    assert len(records) == 1


def test_load_with_run_id_filter(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record(run_id="run1"))
    append_epoch(p, _make_record(run_id="run2"))
    records = load_training_log(p, run_id="run1")
    assert all(r.run_id == "run1" for r in records)
    assert len(records) == 1


def test_missing_file_returns_empty(tmp_path: Path):
    records = load_training_log(tmp_path / "nope.jsonl")
    assert records == []


def test_malformed_line_skipped(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"bad": true}\n')
    append_epoch(p, _make_record())
    records = load_training_log(p)
    assert len(records) == 1


def test_summarize_returns_training_summary(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record())
    records = load_training_log(p)
    s = summarize_training_log(records, "run1")
    assert isinstance(s, TrainingSummary)


def test_best_epoch_has_highest_val_auc(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record(epoch=1, val_auc=0.80))
    append_epoch(p, _make_record(epoch=2, val_auc=0.92))
    append_epoch(p, _make_record(epoch=3, val_auc=0.88))
    records = load_training_log(p)
    s = summarize_training_log(records, "run1")
    assert s.best_epoch == 2
    assert abs(s.best_val_auc - 0.92) < 1e-9


def test_empty_records_returns_empty_flag():
    s = summarize_training_log([], "run1")
    assert s.flag == "EMPTY"


def test_training_run_record_frozen():
    rec = _make_record()
    try:
        rec.epoch = 99  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception:
        pass


def test_multiple_runs_coexist(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    append_epoch(p, _make_record(run_id="A", epoch=1))
    append_epoch(p, _make_record(run_id="B", epoch=1))
    all_records = load_training_log(p)
    assert len(all_records) == 2
    run_a = load_training_log(p, run_id="A")
    run_b = load_training_log(p, run_id="B")
    assert len(run_a) == 1
    assert len(run_b) == 1


def test_format_returns_str():
    s = summarize_training_log([_make_record()], "run1")
    assert isinstance(format_training_summary(s), str)


def test_format_has_best_val_auc():
    s = summarize_training_log([_make_record(val_auc=0.876)], "run1")
    text = format_training_summary(s)
    assert "best_val_auc" in text
