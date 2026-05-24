"""Tests for Skills.cnn_checkpoint_manager (13 tests)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from Skills.cnn_checkpoint_manager import (
    CheckpointRecord,
    CheckpointSummary,
    format_checkpoint_summary,
    list_checkpoints,
    prune_checkpoints,
    select_best,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(directory: Path, records: list[dict]) -> None:
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps({"checkpoints": records}))


def _make_record(epoch: int, val_loss: float, val_auc: float) -> dict:
    return {
        "path": f"/tmp/epoch_{epoch:04d}.pt",
        "epoch": epoch,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "created_at": "2026-05-24T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_nonexistent_dir_returns_invalid(self) -> None:
        result = list_checkpoints(Path("/tmp/does_not_exist_xyz_42"))
        assert result.flag == "INVALID"
        assert result.records == ()

    def test_empty_dir_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_checkpoints(Path(tmpdir))
        assert result.flag == "EMPTY"

    def test_reads_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [_make_record(1, 0.4, 0.8)])
            result = list_checkpoints(Path(tmpdir))
        assert result.flag == "OK"
        assert len(result.records) == 1

    def test_best_by_loss_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [
                _make_record(1, 0.5, 0.7),
                _make_record(2, 0.3, 0.85),
                _make_record(3, 0.4, 0.8),
            ])
            result = list_checkpoints(Path(tmpdir))
        assert result.best_by_loss is not None
        assert result.best_by_loss.epoch == 2

    def test_best_by_auc_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [
                _make_record(1, 0.5, 0.7),
                _make_record(2, 0.3, 0.85),
                _make_record(3, 0.4, 0.8),
            ])
            result = list_checkpoints(Path(tmpdir))
        assert result.best_by_auc is not None
        assert result.best_by_auc.epoch == 2

    def test_fallback_scans_pt_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "epoch_0001.pt").write_bytes(b"")
            (Path(tmpdir) / "epoch_0002.pt").write_bytes(b"")
            result = list_checkpoints(Path(tmpdir))
        assert result.flag == "OK"
        assert len(result.records) == 2

    def test_records_are_checkpoint_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [_make_record(1, 0.4, 0.8)])
            result = list_checkpoints(Path(tmpdir))
        for rec in result.records:
            assert isinstance(rec, CheckpointRecord)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------


class TestSelectBest:
    def test_returns_best_by_auc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [
                _make_record(1, 0.5, 0.7),
                _make_record(2, 0.3, 0.9),
            ])
            summary = list_checkpoints(Path(tmpdir))
        best = select_best(summary, criterion="val_auc")
        assert best is not None
        assert best.epoch == 2

    def test_returns_best_by_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [
                _make_record(1, 0.5, 0.7),
                _make_record(2, 0.3, 0.9),
            ])
            summary = list_checkpoints(Path(tmpdir))
        best = select_best(summary, criterion="val_loss")
        assert best is not None
        assert best.epoch == 2

    def test_invalid_criterion_raises(self) -> None:
        import pytest
        empty_summary = CheckpointSummary(
            checkpoint_dir="/tmp",
            records=(),
            best_by_loss=None,
            best_by_auc=None,
            flag="EMPTY",
        )
        with pytest.raises(ValueError):
            select_best(empty_summary, criterion="bad_criterion")

    def test_empty_summary_returns_none(self) -> None:
        empty_summary = CheckpointSummary(
            checkpoint_dir="/tmp",
            records=(),
            best_by_loss=None,
            best_by_auc=None,
            flag="EMPTY",
        )
        assert select_best(empty_summary) is None


# ---------------------------------------------------------------------------
# prune_checkpoints
# ---------------------------------------------------------------------------


class TestPruneCheckpoints:
    def test_dry_run_returns_to_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [
                _make_record(1, 0.5, 0.7),
                _make_record(2, 0.3, 0.9),
                _make_record(3, 0.4, 0.8),
                _make_record(4, 0.45, 0.75),
            ])
            summary = list_checkpoints(Path(tmpdir))
        to_delete = prune_checkpoints(summary, keep_top_k=2, dry_run=True)
        assert len(to_delete) == 2

    def test_empty_summary_returns_empty(self) -> None:
        empty_summary = CheckpointSummary(
            checkpoint_dir="/tmp",
            records=(),
            best_by_loss=None,
            best_by_auc=None,
            flag="EMPTY",
        )
        assert prune_checkpoints(empty_summary) == ()


# ---------------------------------------------------------------------------
# format_checkpoint_summary
# ---------------------------------------------------------------------------


class TestFormatCheckpointSummary:
    def test_returns_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [_make_record(1, 0.4, 0.8)])
            summary = list_checkpoints(Path(tmpdir))
        out = format_checkpoint_summary(summary)
        assert isinstance(out, str)

    def test_contains_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metrics(Path(tmpdir), [_make_record(1, 0.4, 0.8)])
            summary = list_checkpoints(Path(tmpdir))
        out = format_checkpoint_summary(summary)
        assert "OK" in out
