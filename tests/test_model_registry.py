"""Tests for Skills/model_registry.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

import pytest

from model_registry import (
    RegistryEntry,
    RegistryResult,
    format_registry,
    get_best,
    list_models,
    register,
)


def _make_entry(model_id: str, auc: float = 0.85, brier: float = 0.10) -> RegistryEntry:
    return RegistryEntry(
        model_id=model_id,
        model_type="xgboost",
        model_path=f"/models/{model_id}.xgb",
        auc=auc,
        brier=brier,
        n_train=500,
        registered_at="2026-05-25T00:00:00+00:00",
        notes="test entry",
    )


def test_register_creates_file_on_first_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        assert not path.exists()
        register(path, _make_entry("m1"))
        assert path.exists()


def test_second_register_same_id_raises_value_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("m1"))
        with pytest.raises(ValueError, match="m1"):
            register(path, _make_entry("m1"))


def test_list_models_returns_all_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("m1"))
        register(path, _make_entry("m2"))
        result = list_models(path)
        ids = {e.model_id for e in result.entries}
        assert ids == {"m1", "m2"}


def test_get_best_metric_auc_returns_highest():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("low", auc=0.70))
        register(path, _make_entry("high", auc=0.95))
        best = get_best(path, metric="auc")
        assert best is not None
        assert best.model_id == "high"


def test_get_best_on_empty_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        assert get_best(path) is None


def test_missing_file_returns_empty_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.json"
        result = list_models(path)
        assert result.flag == "EMPTY"


def test_format_returns_str():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("m1"))
        result = list_models(path)
        md = format_registry(result)
        assert isinstance(md, str)


def test_registry_entry_frozen():
    entry = _make_entry("x")
    try:
        entry.model_id = "modified"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_n_models_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        for i in range(3):
            register(path, _make_entry(f"m{i}"))
        result = list_models(path)
        assert result.n_models == 3


def test_notes_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        entry = RegistryEntry(
            model_id="noted",
            model_type="bayesian",
            model_path="/p",
            auc=0.8,
            brier=0.1,
            n_train=100,
            registered_at="2026-01-01T00:00:00Z",
            notes="special run",
        )
        register(path, entry)
        result = list_models(path)
        assert result.entries[0].notes == "special run"


def test_registered_at_non_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("m1"))
        result = list_models(path)
        assert result.entries[0].registered_at != ""


def test_model_type_stored():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        entry = RegistryEntry(
            model_id="cnn1",
            model_type="cnn",
            model_path="/models/cnn1",
            auc=0.9,
            brier=0.09,
            n_train=2000,
            registered_at="2026-05-25T00:00:00Z",
            notes="",
        )
        register(path, entry)
        result = list_models(path)
        assert result.entries[0].model_type == "cnn"


def test_registry_result_frozen():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        register(path, _make_entry("m1"))
        result = list_models(path)
        try:
            result.flag = "MODIFIED"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except Exception as exc:
            assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__
