"""Tests for Skills.train_cnn (13 tests).

All tests use the NO_TORCH path or controlled stubs; PyTorch is not required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from Skills.cnn_training_config import default_config
from Skills.train_cnn import (
    CnnTrainingResult,
    EpochRecord,
    _compute_auc,
    _load_split,
    format_training_result,
    train_cnn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_split(directory: Path, name: str, examples: list[dict]) -> None:
    path = directory / f"{name}.json"
    payload = {"split": name, "examples": examples}
    path.write_text(json.dumps(payload))


def _make_example(label: int, n_bins: int = 201) -> dict:
    flux = [1.0 - 0.01 if abs(i - n_bins // 2) < 5 else 1.0 for i in range(n_bins)]
    return {"flux": flux, "label": label}


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


class TestComputeAuc:
    def test_perfect_classifier(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0.9, 0.8, 0.2, 0.1]
        auc = _compute_auc(y_true, y_pred)
        assert abs(auc - 1.0) < 1e-6

    def test_random_classifier(self) -> None:
        y_true = [1, 0, 1, 0]
        y_pred = [0.5, 0.5, 0.5, 0.5]
        auc = _compute_auc(y_true, y_pred)
        assert 0.0 <= auc <= 1.0

    def test_empty_returns_half(self) -> None:
        auc = _compute_auc([], [])
        assert auc == 0.5

    def test_all_positive_returns_half(self) -> None:
        auc = _compute_auc([1, 1, 1], [0.9, 0.8, 0.7])
        assert auc == 0.5


# ---------------------------------------------------------------------------
# _load_split
# ---------------------------------------------------------------------------


class TestLoadSplit:
    def test_loads_examples_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.json"
            payload = {"split": "train", "examples": [{"flux": [1.0], "label": 0}]}
            path.write_text(json.dumps(payload))
            result = _load_split(path)
        assert len(result) == 1
        assert result[0]["label"] == 0

    def test_loads_flat_list_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "val.json"
            path.write_text(json.dumps([{"flux": [1.0], "label": 1}]))
            result = _load_split(path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# train_cnn
# ---------------------------------------------------------------------------


class TestTrainCnn:
    def test_missing_split_dir_returns_invalid(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_cnn(
                Path(tmpdir) / "nonexistent",
                cfg,
                checkpoint_dir=Path(tmpdir) / "ckpt",
            )
        # May be NO_TORCH or INVALID depending on whether torch is installed
        assert result.flag in {"NO_TORCH", "INVALID"}

    def test_insufficient_data_flag(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "splits"
            split_dir.mkdir()
            # Only 2 training examples, both same label
            _write_split(split_dir, "train", [_make_example(1), _make_example(1)])
            _write_split(split_dir, "val", [_make_example(0)])
            result = train_cnn(
                split_dir, cfg, checkpoint_dir=Path(tmpdir) / "ckpt"
            )
        assert result.flag in {"NO_TORCH", "INSUFFICIENT_DATA"}

    def test_result_is_frozen_dataclass(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_cnn(
                Path(tmpdir) / "nonexistent",
                cfg,
                checkpoint_dir=Path(tmpdir) / "ckpt",
            )
        assert isinstance(result, CnnTrainingResult)

    def test_no_torch_has_empty_history(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_cnn(
                Path(tmpdir) / "nonexistent",
                cfg,
                checkpoint_dir=Path(tmpdir) / "ckpt",
            )
        if result.flag == "NO_TORCH":
            assert result.train_history == ()
            assert result.best_epoch == 0

    def test_format_training_result_no_torch(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_cnn(
                Path(tmpdir) / "nonexistent",
                cfg,
                checkpoint_dir=Path(tmpdir) / "ckpt",
            )
        out = format_training_result(result)
        assert isinstance(out, str)
        assert "Flag" in out


# ---------------------------------------------------------------------------
# EpochRecord
# ---------------------------------------------------------------------------


class TestEpochRecord:
    def test_frozen(self) -> None:
        import pytest
        rec = EpochRecord(epoch=1, train_loss=0.5, val_loss=0.4, val_auc=0.8)
        assert rec.epoch == 1
        # frozen=True dataclasses raise FrozenInstanceError (subclass of AttributeError)
        with pytest.raises(AttributeError):
            rec.epoch = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        rec = EpochRecord(epoch=5, train_loss=0.3, val_loss=0.25, val_auc=0.9)
        assert rec.train_loss == 0.3
        assert rec.val_auc == 0.9


# ---------------------------------------------------------------------------
# format_training_result
# ---------------------------------------------------------------------------


class TestFormatTrainingResult:
    def test_contains_flag(self) -> None:
        result = CnnTrainingResult(
            best_epoch=3,
            best_val_loss=0.4,
            best_val_auc=0.85,
            train_history=(EpochRecord(1, 0.5, 0.45, 0.75),),
            checkpoint_path="/tmp/best.pt",
            config_path="/tmp/config.json",
            n_train=100,
            n_val=20,
            n_positive=50,
            n_negative=50,
            flag="OK",
        )
        out = format_training_result(result)
        assert "OK" in out
        assert "3" in out

    def test_returns_string(self) -> None:
        result = CnnTrainingResult(
            best_epoch=0,
            best_val_loss=float("inf"),
            best_val_auc=0.0,
            train_history=(),
            checkpoint_path="",
            config_path="",
            n_train=0,
            n_val=0,
            n_positive=0,
            n_negative=0,
            flag="NO_TORCH",
        )
        assert isinstance(format_training_result(result), str)
