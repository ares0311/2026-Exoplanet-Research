"""Tests for Skills.cnn_training_config (13 tests)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from Skills.cnn_training_config import (
    CnnConfigValidation,
    CnnTrainingConfig,
    ConvLayerSpec,
    _config_from_dict,
    _config_to_dict,
    default_config,
    load_config,
    save_config,
    validate_config,
)


class TestDefaultConfig:
    def test_returns_cnn_training_config(self) -> None:
        cfg = default_config()
        assert isinstance(cfg, CnnTrainingConfig)

    def test_n_bins(self) -> None:
        cfg = default_config()
        assert cfg.n_bins == 201

    def test_three_conv_layers(self) -> None:
        cfg = default_config()
        assert len(cfg.conv_layers) == 3

    def test_conv_layer_types(self) -> None:
        cfg = default_config()
        for cl in cfg.conv_layers:
            assert isinstance(cl, ConvLayerSpec)

    def test_default_optimizer(self) -> None:
        cfg = default_config()
        assert cfg.optimizer == "adamw"

    def test_default_dropout(self) -> None:
        cfg = default_config()
        assert cfg.dropout_rate == 0.5
        assert cfg.dense_dropout_rates == (0.5, 0.3)

    def test_default_matches_cnn_spec(self) -> None:
        cfg = default_config()
        assert cfg.dense_units == (256, 64)
        assert cfg.batch_size == 64
        assert cfg.max_epochs == 50

    def test_new_fields_default_to_off(self) -> None:
        cfg = default_config()
        assert cfg.augmentation_flip is False
        assert cfg.augmentation_shift_bins == 0
        assert cfg.use_batch_norm is False


class TestValidateConfig:
    def test_valid_default_ok(self) -> None:
        result = validate_config(default_config())
        assert isinstance(result, CnnConfigValidation)
        assert result.ok is True
        assert result.errors == ()

    def test_invalid_n_bins(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), n_bins=0)
        result = validate_config(cfg)
        assert result.ok is False
        assert any("n_bins" in e for e in result.errors)

    def test_invalid_dropout(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), dropout_rate=1.5)
        result = validate_config(cfg)
        assert result.ok is False
        assert any("dropout_rate" in e for e in result.errors)

    def test_dense_dropout_count_must_match_dense_layers(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), dense_dropout_rates=(0.5,))
        result = validate_config(cfg)
        assert result.ok is False
        assert any("one value per dense layer" in error for error in result.errors)

    def test_invalid_optimizer(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), optimizer="rmsprop")
        result = validate_config(cfg)
        assert result.ok is False
        assert any("optimizer" in e for e in result.errors)

    def test_invalid_even_kernel_size(self) -> None:
        bad_conv = (
            ConvLayerSpec(out_channels=16, kernel_size=4, pool_size=2),
        )
        import dataclasses
        cfg = dataclasses.replace(default_config(), conv_layers=bad_conv)
        result = validate_config(cfg)
        assert result.ok is False
        assert any("kernel_size" in e for e in result.errors)

    def test_invalid_pool_size(self) -> None:
        bad_conv = (
            ConvLayerSpec(out_channels=16, kernel_size=5, pool_size=0),
        )
        import dataclasses
        cfg = dataclasses.replace(default_config(), conv_layers=bad_conv)
        result = validate_config(cfg)
        assert result.ok is False
        assert any("pool_size" in e for e in result.errors)

    def test_invalid_lr(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), learning_rate=-0.01)
        result = validate_config(cfg)
        assert result.ok is False

    def test_invalid_shift_bins(self) -> None:
        import dataclasses
        cfg = dataclasses.replace(default_config(), augmentation_shift_bins=-1)
        result = validate_config(cfg)
        assert result.ok is False
        assert any("shift_bins" in e for e in result.errors)


class TestSerialisation:
    def test_round_trip(self) -> None:
        cfg = default_config()
        d = _config_to_dict(cfg)
        cfg2 = _config_from_dict(d)
        assert cfg == cfg2

    def test_save_load(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(cfg, path)
            assert path.exists()
            loaded = load_config(path)
        assert loaded == cfg

    def test_save_creates_parent_dir(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "config.json"
            save_config(cfg, path)
            assert path.exists()

    def test_json_has_expected_keys(self) -> None:
        cfg = default_config()
        d = _config_to_dict(cfg)
        for key in (
            "n_bins",
            "conv_layers",
            "dense_units",
            "dropout_rate",
            "dense_dropout_rates",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "selection_metric",
            "lr_scheduler_patience",
            "lr_scheduler_factor",
            "min_learning_rate",
            "gradient_clip_norm",
            "augment",
            "augmentation_noise_fraction",
            "augmentation_scale_min",
            "augmentation_scale_max",
            "augmentation_flip",
            "augmentation_shift_bins",
            "use_batch_norm",
            "seed",
            "checkpoint_dir",
        ):
            assert key in d

    def test_old_config_uses_legacy_dropout_for_each_dense_layer(self) -> None:
        payload = _config_to_dict(default_config())
        for key in (
            "dense_dropout_rates",
            "weight_decay",
            "selection_metric",
            "lr_scheduler_patience",
            "lr_scheduler_factor",
            "min_learning_rate",
            "gradient_clip_norm",
            "augmentation_noise_fraction",
            "augmentation_scale_min",
            "augmentation_flip",
            "augmentation_shift_bins",
            "use_batch_norm",
            "augmentation_scale_max",
        ):
            payload.pop(key)

        config = _config_from_dict(payload)

        assert config.dense_dropout_rates == (0.5, 0.5)
        assert config.weight_decay == 0.0
        assert config.selection_metric == "val_loss"
        assert config.augmentation_flip is False
        assert config.augmentation_shift_bins == 0
        assert config.use_batch_norm is False

    def test_conv_layers_serialised_as_list_of_dicts(self) -> None:
        cfg = default_config()
        d = _config_to_dict(cfg)
        assert isinstance(d["conv_layers"], list)
        for cl in d["conv_layers"]:
            assert isinstance(cl, dict)
            assert "out_channels" in cl
