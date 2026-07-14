"""Tests for the bounded external-baseline inference smoke."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from smoke_representation_baseline_inference import (
    cache_pinned_model,
    format_status,
    load_source_gate,
    prepare_relative_magnitude,
    run_model_inference,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packages": [{"wheel": {"size_bytes": 10}}],
        "models": [
            {
                "name": "Chronos-Bolt tiny",
                "role": "general_time_series_foundation_baseline",
                "repo": "example/chronos",
                "commit": "a" * 40,
                "filename": "chronos.onnx",
                "size_bytes": 20,
                "sha256": "b" * 64,
                "embedding_dimension": 256,
            },
            {
                "name": "Astromer2",
                "role": "astronomy_native_foundation_comparator",
                "repo": "example/astromer",
                "commit": "c" * 40,
                "filename": "astromer.onnx",
                "size_bytes": 30,
                "sha256": "d" * 64,
                "embedding_dimension": 256,
            },
        ],
        "aggregate_download_bytes": 60,
    }


def _write_source_gate(tmp_path: Path) -> tuple[Path, Path]:
    contract_path = tmp_path / "contract.json"
    evidence_path = tmp_path / "evidence.json"
    contract_path.write_text(json.dumps(_source_contract()), encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "success",
                "payload_bytes_downloaded": 0,
                "training_authorized": False,
                "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                "aggregate_download_bytes_if_installed": 60,
                "models": [{}, {}],
            }
        ),
        encoding="utf-8",
    )
    return contract_path, evidence_path


def test_load_source_gate_accepts_matching_success_evidence(tmp_path: Path) -> None:
    contract_path, evidence_path = _write_source_gate(tmp_path)
    contract, evidence = load_source_gate(contract_path, evidence_path)
    assert contract["aggregate_download_bytes"] == 60
    assert evidence["status"] == "success"


def test_load_source_gate_fails_on_contract_drift(tmp_path: Path) -> None:
    contract_path, evidence_path = _write_source_gate(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["packages"][0]["wheel"]["size_bytes"] = 11
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="source contract SHA-256 mismatch"):
        load_source_gate(contract_path, evidence_path)


def test_prepare_relative_magnitude_filters_sorts_and_bounds() -> None:
    times = np.arange(2200, dtype=np.float64)[::-1]
    flux = np.linspace(900.0, 1100.0, 2200)
    quality = np.zeros(2200, dtype=np.int32)
    quality[5] = 1
    flux[10] = np.nan
    flux[15] = -1.0
    selected_times, magnitude = prepare_relative_magnitude(times, flux, quality)
    assert selected_times.shape == (2048,)
    assert magnitude.shape == (2048,)
    assert np.all(np.diff(selected_times) > 0)
    assert np.all(np.isfinite(magnitude))
    assert magnitude.dtype == np.float32


def test_prepare_relative_magnitude_rejects_too_few_clean_cadences() -> None:
    with pytest.raises(ValueError, match="clean positive-flux cadences"):
        prepare_relative_magnitude(
            np.arange(199),
            np.ones(199),
            np.zeros(199, dtype=np.int32),
        )


def test_cache_pinned_model_uses_exact_revision_and_verifies_hash(tmp_path: Path) -> None:
    payload = b"pinned-onnx"
    model = {
        "repo": "example/model",
        "commit": "a" * 40,
        "filename": "model.onnx",
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }
    calls: list[dict[str, Any]] = []

    def _download(**kwargs: Any) -> str:
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(payload)
        return str(path)

    path, downloaded = cache_pinned_model(model, tmp_path, downloader=_download)
    assert downloaded is True
    assert path.read_bytes() == payload
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["force_download"] is True
    assert Path(calls[0]["local_dir"]).is_relative_to(tmp_path)


def test_cache_pinned_model_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    model = {
        "repo": "example/model",
        "commit": "a" * 40,
        "filename": "model.onnx",
        "size_bytes": 3,
        "sha256": _sha256_bytes(b"good"),
    }

    def _download(**kwargs: Any) -> str:
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(b"bad")
        return str(path)

    with pytest.raises(ValueError, match="downloaded model mismatch"):
        cache_pinned_model(model, tmp_path, downloader=_download)


class _FakeSessionOptions:
    intra_op_num_threads = 0
    inter_op_num_threads = 0


class _FakeSession:
    disabled = False

    def disable_fallback(self) -> None:
        self.disabled = True


class _FakeOrt:
    last_options: _FakeSessionOptions | None = None
    last_providers: list[str] | None = None

    @classmethod
    def SessionOptions(cls) -> _FakeSessionOptions:
        return _FakeSessionOptions()

    @classmethod
    def InferenceSession(
        cls,
        _path: str,
        *,
        sess_options: _FakeSessionOptions,
        providers: list[str],
    ) -> _FakeSession:
        cls.last_options = sess_options
        cls.last_providers = providers
        return _FakeSession()


class _FakeChronos:
    call_arity = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __call__(self, magnitude: np.ndarray) -> np.ndarray:
        type(self).call_arity = 1
        assert magnitude.shape == (256,)
        return np.ones((1, 1, 1, 256), dtype=np.float32)


class _FakeAstromer:
    call_arity = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __call__(self, times: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        type(self).call_arity = 2
        assert times.shape == magnitude.shape == (256,)
        return np.full((1, 1, 1, 256), 2.0, dtype=np.float32)


@pytest.mark.parametrize(
    ("role", "expected_arity"),
    [
        ("general_time_series_foundation_baseline", 1),
        ("astronomy_native_foundation_comparator", 2),
    ],
)
def test_run_model_inference_uses_bounded_cpu_session_and_correct_inputs(
    tmp_path: Path,
    role: str,
    expected_arity: int,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    model = {
        "name": role,
        "role": role,
        "repo": "example/model",
        "commit": "a" * 40,
        "filename": "model.onnx",
        "embedding_dimension": 256,
    }
    result = run_model_inference(
        model,
        model_path,
        np.arange(256, dtype=np.float64),
        np.ones(256, dtype=np.float32),
        ort_module=_FakeOrt,
        chronos_class=_FakeChronos,
        astromer_class=_FakeAstromer,
    )
    assert result["status"] == "success"
    assert result["embedding_shape"] == [1, 1, 1, 256]
    assert result["finite"] is True
    assert _FakeOrt.last_options is not None
    assert _FakeOrt.last_options.intra_op_num_threads == 1
    assert _FakeOrt.last_options.inter_op_num_threads == 1
    assert _FakeOrt.last_providers == ["CPUExecutionProvider"]
    actual_arity = _FakeChronos.call_arity if expected_arity == 1 else _FakeAstromer.call_arity
    assert actual_arity == expected_arity


def test_format_status_preserves_training_guardrail() -> None:
    text = format_status(
        {
            "status": "success",
            "results": [{}, {}],
            "model_cache_bytes": 30,
            "downloaded_this_run_bytes": 30,
            "training_authorized": False,
        }
    )
    assert "models=2" in text
    assert "training_authorized=False" in text
