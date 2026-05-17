"""Tests for Skills.config_manager."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.config_manager import (
    PipelineConfig,
    default_config,
    load_config,
    validate_config,
)


def _write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    return p


class TestDefaultConfig:
    def test_returns_pipeline_config(self) -> None:
        cfg = default_config()
        assert isinstance(cfg, PipelineConfig)

    def test_mission_default_tess(self) -> None:
        cfg = default_config()
        assert cfg["mission"] == "TESS"

    def test_scorer_default_bayesian(self) -> None:
        cfg = default_config()
        assert cfg["scorer"] == "bayesian"

    def test_min_snr_positive(self) -> None:
        cfg = default_config()
        assert cfg["min_snr"] > 0


class TestLoadConfig:
    def test_loads_json_file(self, tmp_path: Path) -> None:
        p = _write_json(tmp_path, {"mission": "TESS", "min_snr": 7.0})
        cfg = load_config(p)
        assert cfg["min_snr"] == pytest.approx(7.0)

    def test_overrides_defaults(self, tmp_path: Path) -> None:
        p = _write_json(tmp_path, {"scorer": "xgboost"})
        cfg = load_config(p)
        assert cfg["scorer"] == "xgboost"

    def test_missing_keys_use_defaults(self, tmp_path: Path) -> None:
        p = _write_json(tmp_path, {})
        cfg = load_config(p)
        assert cfg["mission"] == "TESS"

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MIN_SNR", "9.5")
        p = _write_json(tmp_path, {})
        cfg = load_config(p)
        assert cfg["min_snr"] == pytest.approx(9.5)

    def test_attribute_access(self, tmp_path: Path) -> None:
        p = _write_json(tmp_path, {"scorer": "ensemble"})
        cfg = load_config(p)
        assert cfg.scorer == "ensemble"

    def test_get_with_default(self, tmp_path: Path) -> None:
        p = _write_json(tmp_path, {})
        cfg = load_config(p)
        assert cfg.get("nonexistent_key", 42) == 42


class TestValidateConfig:
    def test_valid_config_returns_empty_list(self) -> None:
        errors = validate_config(default_config())
        assert errors == []

    def test_invalid_mission_flagged(self) -> None:
        cfg = PipelineConfig({"mission": "Hubble", "min_snr": 5.0, "scorer": "bayesian"})
        errors = validate_config(cfg)
        assert any("mission" in e.lower() for e in errors)

    def test_invalid_scorer_flagged(self) -> None:
        cfg = PipelineConfig({"mission": "TESS", "min_snr": 5.0, "scorer": "deep_learning"})
        errors = validate_config(cfg)
        assert any("scorer" in e.lower() for e in errors)

    def test_out_of_range_snr_flagged(self) -> None:
        cfg = PipelineConfig({"mission": "TESS", "min_snr": 999.0, "scorer": "bayesian"})
        errors = validate_config(cfg)
        assert any("min_snr" in e for e in errors)

    def test_tmag_min_ge_max_flagged(self) -> None:
        cfg = PipelineConfig({
            "mission": "TESS", "min_snr": 5.0, "scorer": "bayesian",
            "tmag_min": 14.0, "tmag_max": 10.0,
        })
        errors = validate_config(cfg)
        assert any("tmag" in e for e in errors)

    def test_missing_required_key_flagged(self) -> None:
        cfg = PipelineConfig({"mission": "TESS", "scorer": "bayesian"})
        errors = validate_config(cfg)
        assert any("min_snr" in e for e in errors)
