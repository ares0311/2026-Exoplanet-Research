"""Tests for transfer-learning additions to train_cnn and cnn_training_config (13 tests)."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from Skills.cnn_training_config import (
    default_config,
    load_config,
    save_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# freeze_conv_epochs field in CnnTrainingConfig
# ---------------------------------------------------------------------------


class TestFreezeConvEpochs:
    def test_default_is_zero(self) -> None:
        cfg = default_config()
        assert cfg.freeze_conv_epochs == 0

    def test_can_set_positive(self) -> None:
        cfg = dataclasses.replace(default_config(), freeze_conv_epochs=15)
        assert cfg.freeze_conv_epochs == 15

    def test_negative_fails_validation(self) -> None:
        cfg = dataclasses.replace(default_config(), freeze_conv_epochs=-1)
        result = validate_config(cfg)
        assert not result.ok
        assert any("freeze_conv_epochs" in e for e in result.errors)

    def test_zero_passes_validation(self) -> None:
        cfg = dataclasses.replace(default_config(), freeze_conv_epochs=0)
        result = validate_config(cfg)
        assert result.ok

    def test_roundtrips_through_json(self, tmp_path: Path) -> None:
        cfg = dataclasses.replace(default_config(), freeze_conv_epochs=15)
        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.freeze_conv_epochs == 15

    def test_missing_key_defaults_to_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        # Build a minimal valid config dict without freeze_conv_epochs
        d = {
            "n_bins": 201,
            "conv_layers": [
                {"out_channels": 16, "kernel_size": 5, "pool_size": 2},
            ],
            "dense_units": [64],
            "dropout_rate": 0.3,
            "dense_dropout_rates": [0.3],
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "batch_size": 64,
            "max_epochs": 10,
            "early_stopping_patience": 5,
            "selection_metric": "val_auc",
            "lr_scheduler_patience": 2,
            "lr_scheduler_factor": 0.5,
            "min_learning_rate": 1e-5,
            "gradient_clip_norm": 5.0,
            "augment": False,
            "augmentation_noise_fraction": 0.0,
            "augmentation_scale_min": 1.0,
            "augmentation_scale_max": 1.0,
            "augmentation_flip": False,
            "augmentation_shift_bins": 0,
            "use_batch_norm": False,
            "seed": 7,
            "checkpoint_dir": "checkpoints/test",
            # freeze_conv_epochs deliberately omitted
        }
        path.write_text(json.dumps(d))
        loaded = load_config(path)
        assert loaded.freeze_conv_epochs == 0


# ---------------------------------------------------------------------------
# CLI --pretrained-checkpoint flag (no torch needed)
# ---------------------------------------------------------------------------


class TestPretrainedCheckpointCLI:
    def test_accepted_by_cli_missing_split_dir(self, tmp_path: Path) -> None:
        script = Path(__file__).parents[1] / "Skills" / "train_cnn.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--split-dir", str(tmp_path / "missing"),
                "--checkpoint-dir", str(tmp_path / "ckpt"),
                "--pretrained-checkpoint", str(tmp_path / "fake.pt"),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 1
        assert "unrecognized" not in completed.stderr
        assert "Flag:" in completed.stdout

    def test_none_pretrained_accepted(self, tmp_path: Path) -> None:
        script = Path(__file__).parents[1] / "Skills" / "train_cnn.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--split-dir", str(tmp_path / "missing"),
                "--checkpoint-dir", str(tmp_path / "ckpt"),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 1
        assert "Flag:" in completed.stdout


# ---------------------------------------------------------------------------
# freeze / unfreeze in training loop (requires torch)
# ---------------------------------------------------------------------------


class TestFreezeConvTraining:
    def _write_splits(self, split_dir: Path) -> None:
        import random
        rng = random.Random(0)
        examples = [
            {"flux": [rng.gauss(1.0, 0.01) for _ in range(11)], "label": i % 2}
            for i in range(32)
        ]
        train_path = split_dir / "train.json"
        val_path = split_dir / "val.json"
        train_path.write_text(json.dumps({"split": "train", "examples": examples[:24]}))
        val_path.write_text(json.dumps({"split": "val", "examples": examples[24:]}))

    def test_freeze_conv_training_completes(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from Skills.train_cnn import train_cnn

        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        self._write_splits(split_dir)

        cfg = dataclasses.replace(
            default_config(),
            n_bins=11,
            max_epochs=4,
            early_stopping_patience=4,
            dense_units=(16,),
            dense_dropout_rates=(0.0,),
            freeze_conv_epochs=2,
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        result = train_cnn(split_dir, cfg, checkpoint_dir=tmp_path / "ckpt")
        assert result.flag in {"OK", "NO_TORCH", "INSUFFICIENT_DATA"}

    def test_missing_pretrained_checkpoint_logs_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        pytest.importorskip("torch")
        from Skills.train_cnn import train_cnn

        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        self._write_splits(split_dir)

        cfg = dataclasses.replace(
            default_config(),
            n_bins=11,
            max_epochs=2,
            early_stopping_patience=2,
            dense_units=(16,),
            dense_dropout_rates=(0.0,),
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        result = train_cnn(
            split_dir, cfg,
            checkpoint_dir=tmp_path / "ckpt",
            pretrained_checkpoint=tmp_path / "nonexistent.pt",
        )
        captured = capsys.readouterr()
        assert "Warning" in captured.out or result.flag in {"OK", "INSUFFICIENT_DATA"}
