"""Tests for Skills/cnn_model_config.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from cnn_model_config import (
    CnnModelConfig,
    ModelConfigValidation,
    default_model_config,
    format_model_config,
    model_config_from_dict,
    model_config_to_dict,
    validate_model_config,
)


def test_default_returns_cnn_model_config():
    c = default_model_config()
    assert isinstance(c, CnnModelConfig)


def test_validate_passes_default():
    c = default_model_config()
    v = validate_model_config(c)
    assert v.ok
    assert v.flag == "OK"


def test_even_kernel_size_fails():
    c = CnnModelConfig(
        n_bins=201, n_filters=(16, 32), kernel_size=4, n_layers=2,
        dropout=0.3, dense_units=(64,), flag="OK"
    )
    v = validate_model_config(c)
    assert not v.ok
    assert v.flag == "INVALID"


def test_dropout_ge_1_fails():
    c = CnnModelConfig(
        n_bins=201, n_filters=(16, 32), kernel_size=5, n_layers=2,
        dropout=1.0, dense_units=(64,), flag="OK"
    )
    v = validate_model_config(c)
    assert not v.ok


def test_n_bins_le_0_fails():
    c = CnnModelConfig(
        n_bins=0, n_filters=(16,), kernel_size=5, n_layers=1,
        dropout=0.3, dense_units=(64,), flag="OK"
    )
    v = validate_model_config(c)
    assert not v.ok


def test_n_layers_zero_fails():
    c = CnnModelConfig(
        n_bins=201, n_filters=(), kernel_size=5, n_layers=0,
        dropout=0.3, dense_units=(64,), flag="OK"
    )
    v = validate_model_config(c)
    assert not v.ok


def test_n_filters_len_mismatch_fails():
    c = CnnModelConfig(
        n_bins=201, n_filters=(16, 32), kernel_size=5, n_layers=3,
        dropout=0.3, dense_units=(64,), flag="OK"
    )
    v = validate_model_config(c)
    assert not v.ok


def test_json_round_trip():
    c = default_model_config()
    d = model_config_to_dict(c)
    c2 = model_config_from_dict(d)
    assert c2.n_bins == c.n_bins
    assert c2.kernel_size == c.kernel_size
    assert c2.dropout == c.dropout


def test_format_has_n_bins():
    c = default_model_config()
    text = format_model_config(c)
    assert "n_bins" in text


def test_flag_is_ok_on_valid():
    c = default_model_config()
    assert c.flag == "OK"


def test_model_config_validation_ok_true_on_valid():
    c = default_model_config()
    v = validate_model_config(c)
    assert isinstance(v, ModelConfigValidation)
    assert v.ok is True


def test_n_filters_is_tuple_after_round_trip():
    c = default_model_config()
    d = model_config_to_dict(c)
    c2 = model_config_from_dict(d)
    assert isinstance(c2.n_filters, tuple)


def test_cli_show_returns_0():
    from cnn_model_config import _cli
    rc = _cli(["show"])
    assert rc == 0
