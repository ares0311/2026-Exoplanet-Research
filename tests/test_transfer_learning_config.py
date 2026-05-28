"""Tests for Skills/transfer_learning_config.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transfer_learning_config import (
    TransferConfig,
    default_transfer_config,
    format_transfer_config,
    load_transfer_config,
    save_transfer_config,
    validate_transfer_config,
)


def test_default_returns_transfer_config():
    cfg = default_transfer_config()
    assert isinstance(cfg, TransferConfig)


def test_default_frozen_layers_nonempty():
    cfg = default_transfer_config()
    assert len(cfg.frozen_layers) > 0


def test_default_valid():
    cfg = default_transfer_config()
    errors = validate_transfer_config(cfg)
    assert errors == []


def test_invalid_learning_rate():
    cfg = default_transfer_config()
    bad = TransferConfig(
        frozen_layers=cfg.frozen_layers, learning_rate=-0.001,
        fine_tune_lr_multiplier=cfg.fine_tune_lr_multiplier,
        n_fine_tune_epochs=cfg.n_fine_tune_epochs,
        n_warmup_epochs=cfg.n_warmup_epochs, weight_decay=cfg.weight_decay,
        dropout_rate=cfg.dropout_rate, notes="",
    )
    errors = validate_transfer_config(bad)
    assert any("learning_rate" in e for e in errors)


def test_invalid_dropout():
    cfg = default_transfer_config()
    bad = TransferConfig(
        frozen_layers=cfg.frozen_layers, learning_rate=cfg.learning_rate,
        fine_tune_lr_multiplier=cfg.fine_tune_lr_multiplier,
        n_fine_tune_epochs=cfg.n_fine_tune_epochs,
        n_warmup_epochs=cfg.n_warmup_epochs, weight_decay=cfg.weight_decay,
        dropout_rate=1.5, notes="",
    )
    errors = validate_transfer_config(bad)
    assert any("dropout" in e for e in errors)


def test_save_and_load_roundtrip(tmp_path):
    cfg = default_transfer_config()
    p = tmp_path / "transfer.json"
    save_transfer_config(cfg, p)
    loaded = load_transfer_config(p)
    assert loaded.learning_rate == cfg.learning_rate
    assert loaded.frozen_layers == cfg.frozen_layers


def test_save_creates_file(tmp_path):
    cfg = default_transfer_config()
    p = tmp_path / "sub" / "transfer.json"
    save_transfer_config(cfg, p)
    assert p.exists()


def test_format_returns_string():
    cfg = default_transfer_config()
    s = format_transfer_config(cfg)
    assert isinstance(s, str)
    assert "Transfer" in s


def test_format_shows_valid():
    cfg = default_transfer_config()
    s = format_transfer_config(cfg)
    assert "VALID" in s


def test_config_frozen():
    cfg = default_transfer_config()
    try:
        cfg.learning_rate = 1.0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_fine_tune_lr_multiplier_range():
    cfg = default_transfer_config()
    bad = TransferConfig(
        frozen_layers=cfg.frozen_layers, learning_rate=cfg.learning_rate,
        fine_tune_lr_multiplier=2.0,
        n_fine_tune_epochs=cfg.n_fine_tune_epochs,
        n_warmup_epochs=cfg.n_warmup_epochs, weight_decay=cfg.weight_decay,
        dropout_rate=cfg.dropout_rate, notes="",
    )
    errors = validate_transfer_config(bad)
    assert any("fine_tune_lr_multiplier" in e for e in errors)


def test_notes_preserved(tmp_path):
    cfg = default_transfer_config()
    cfg2 = TransferConfig(
        frozen_layers=cfg.frozen_layers, learning_rate=cfg.learning_rate,
        fine_tune_lr_multiplier=cfg.fine_tune_lr_multiplier,
        n_fine_tune_epochs=cfg.n_fine_tune_epochs,
        n_warmup_epochs=cfg.n_warmup_epochs, weight_decay=cfg.weight_decay,
        dropout_rate=cfg.dropout_rate, notes="custom note",
    )
    p = tmp_path / "cfg.json"
    save_transfer_config(cfg2, p)
    loaded = load_transfer_config(p)
    assert "custom note" in loaded.notes
