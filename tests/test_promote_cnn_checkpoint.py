"""Tests for Skills/promote_cnn_checkpoint.py (offline only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.promote_cnn_checkpoint import (
    PromotionResult,
    _sha256_file,
    format_promotion_result,
    promote_cnn_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_calibration(path: Path, *, flag: str = "OK", auc: float = 0.88) -> None:
    path.write_text(json.dumps({
        "method": "platt",
        "platt_a": 1.23,
        "platt_b": -0.45,
        "n_val_samples": 148,
        "fitted_at": "2026-06-11T10:00:00+00:00",
        "gate_auc": 0.85,
        "gate_f1": 0.80,
        "test_auc_raw": auc,
        "test_f1_cal": 0.83,
        "test_brier_cal": 0.12,
        "test_ece_cal": 0.04,
        "flag": flag,
    }))


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-checkpoint-data-for-testing")


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------


class TestSha256File:
    def test_returns_hex_string(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        result = _sha256_file(p)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        assert _sha256_file(p) == _sha256_file(p)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert _sha256_file(a) != _sha256_file(b)


# ---------------------------------------------------------------------------
# promote_cnn_checkpoint
# ---------------------------------------------------------------------------


class TestPromoteCnnCheckpoint:
    def test_missing_checkpoint_returns_missing_flag(self, tmp_path: Path) -> None:
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        result = promote_cnn_checkpoint(
            tmp_path / "missing.pt", cal, tmp_path / "registry.json"
        )
        assert result.flag == "MISSING_FILE"
        assert not result.model_id

    def test_missing_calibration_returns_missing_flag(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        result = promote_cnn_checkpoint(
            ckpt, tmp_path / "missing_cal.json", tmp_path / "registry.json"
        )
        assert result.flag == "MISSING_FILE"

    def test_gates_not_met_returns_gates_flag(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal, flag="FAIL")
        result = promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json"
        )
        assert result.flag == "GATES_NOT_MET"

    def test_successful_promotion(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        result = promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json"
        )
        assert result.flag == "PROMOTED"

    def test_model_id_starts_with_cnn(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        result = promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json"
        )
        assert result.model_id.startswith("cnn_")

    def test_sha256_is_hex_64_chars(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        result = promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json"
        )
        assert len(result.sha256) == 64

    def test_metrics_extracted(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal, auc=0.91)
        result = promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json"
        )
        assert result.auc == pytest.approx(0.91)
        assert result.f1 == pytest.approx(0.83)
        assert result.platt_a == pytest.approx(1.23)
        assert result.platt_b == pytest.approx(-0.45)

    def test_registry_json_created(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        reg = tmp_path / "registry.json"
        promote_cnn_checkpoint(ckpt, cal, reg)
        assert reg.exists()

    def test_manifest_written(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        manifest = tmp_path / "manifest.json"
        promote_cnn_checkpoint(
            ckpt, cal, tmp_path / "registry.json", manifest_path=manifest
        )
        assert manifest.exists()
        m = json.loads(manifest.read_text())
        assert m["flag"] == "PROMOTED"
        assert "sha256" in m

    def test_default_manifest_path(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "cnn" / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        promote_cnn_checkpoint(ckpt, cal, tmp_path / "registry.json")
        assert (tmp_path / "cnn" / "promotion_manifest.json").exists()

    def test_already_registered_is_idempotent(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        reg = tmp_path / "registry.json"
        result1 = promote_cnn_checkpoint(ckpt, cal, reg)
        assert result1.flag == "PROMOTED"
        result2 = promote_cnn_checkpoint(ckpt, cal, reg)
        assert result2.flag == "ALREADY_REGISTERED"


# ---------------------------------------------------------------------------
# format_promotion_result
# ---------------------------------------------------------------------------


class TestFormatPromotionResult:
    def test_format_promoted_contains_sha256(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "best.pt"
        _write_checkpoint(ckpt)
        cal = tmp_path / "calibration.json"
        _write_calibration(cal)
        result = promote_cnn_checkpoint(ckpt, cal, tmp_path / "registry.json")
        text = format_promotion_result(result)
        assert result.sha256 in text
        assert "PRODUCTION_READINESS" in text
        assert "git commit" in text

    def test_format_missing_file(self) -> None:
        result = PromotionResult(
            model_id="", sha256="", auc=None, f1=None, brier=None, ece=None,
            platt_a=None, platt_b=None,
            registry_path="models/registry.json",
            manifest_path="models/cnn/manifest.json",
            promoted_at="2026-06-11T10:00:00+00:00",
            flag="MISSING_FILE",
        )
        text = format_promotion_result(result)
        assert "MISSING_FILE" in text

    def test_format_gates_not_met(self) -> None:
        result = PromotionResult(
            model_id="", sha256="", auc=0.72, f1=0.61, brier=0.21, ece=0.09,
            platt_a=1.0, platt_b=0.0,
            registry_path="models/registry.json",
            manifest_path="models/cnn/manifest.json",
            promoted_at="2026-06-11T10:00:00+00:00",
            flag="GATES_NOT_MET",
        )
        text = format_promotion_result(result)
        assert "GATES_NOT_MET" in text
