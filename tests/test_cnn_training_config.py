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

    def test_two_conv_layers(self) -> None:
        cfg = default_config()
        assert len(cfg.conv_layers) == 2

    def test_conv_layer_types(self) -> None:
        cfg = default_config()
        for cl in cfg.conv_layers:
            assert isinstance(cl, ConvLayerSpec)

    def test_default_optimizer(self) -> None:
        cfg = default_config()
        assert cfg.optimizer == "adam"

    def test_default_dropout(self) -> None:
        cfg = default_config()
        assert cfg.dropout_rate == 0.5


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
        for key in ("n_bins", "conv_layers", "dense_units", "dropout_rate",
                    "optimizer", "learning_rate", "batch_size", "max_epochs",
                    "early_stopping_patience", "augment", "seed", "checkpoint_dir"):
            assert key in d

    def test_conv_layers_serialised_as_list_of_dicts(self) -> None:
        cfg = default_config()
        d = _config_to_dict(cfg)
        assert isinstance(d["conv_layers"], list)
        for cl in d["conv_layers"]:
            assert isinstance(cl, dict)
            assert "out_channels" in cl
