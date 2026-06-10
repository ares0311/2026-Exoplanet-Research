"""Tests for Skills.cnn_split_validator."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.build_cnn_training_data import load_training_examples, write_training_splits
from Skills.cnn_split_validator import (
    _cli,
    format_validation_summary,
    validate_split_dir,
    validate_split_manifest,
)


def _snippet(
    *,
    tic_id: int,
    label: int,
    example_id: str | None = None,
    n_bins: int = 5,
) -> dict:
    return {
        "example_id": example_id or f"tic-{tic_id}-{label}",
        "tic_id": tic_id,
        "label": label,
        "phase": [round(-0.5 + (idx + 0.5) / n_bins, 4) for idx in range(n_bins)],
        "flux": [0.99 if idx == n_bins // 2 else 1.0 for idx in range(n_bins)],
        "source": "local-fixture",
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _valid_split_dir(tmp_path: Path) -> Path:
    rows = [
        _snippet(tic_id=100 * label + idx, label=label, example_id=f"{label}-{idx}")
        for label in (0, 1)
        for idx in range(5)
    ]
    input_path = _write_json(tmp_path / "dataset.json", {"snippets": rows})
    examples = load_training_examples([input_path])
    write_training_splits(
        examples,
        tmp_path / "splits",
        created_at="2026-05-20T00:00:00+00:00",
    )
    return tmp_path / "splits"


def _read_manifest(split_dir: Path) -> dict:
    return json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(split_dir: Path, manifest: dict) -> None:
    _write_json(split_dir / "manifest.json", manifest)


def _split_path(split_dir: Path, manifest: dict, split: str) -> Path:
    path = Path(manifest["split_files"][split])
    return path if path.is_absolute() else split_dir / path


def _split_payload(split_dir: Path, split: str) -> dict:
    manifest = _read_manifest(split_dir)
    return json.loads(_split_path(split_dir, manifest, split).read_text(encoding="utf-8"))


def _write_split_payload(split_dir: Path, split: str, payload: dict) -> None:
    manifest = _read_manifest(split_dir)
    _write_json(_split_path(split_dir, manifest, split), payload)


def _codes(split_dir: Path) -> set[str]:
    return {issue.code for issue in validate_split_dir(split_dir).issues}


def test_valid_split_dir_passes(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    result = validate_split_dir(split_dir)
    assert result.ok
    assert result.split_counts["train"] > result.split_counts["val"]
    assert not result.issues


def test_valid_manifest_path_passes_and_formats_summary(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    result = validate_split_manifest(split_dir / "manifest.json")
    summary = format_validation_summary(result)
    assert result.ok
    assert "Status: PASS" in summary
    assert "Train/val/test:" in summary


def test_missing_manifest_reports_error(tmp_path: Path) -> None:
    result = validate_split_dir(tmp_path / "missing")
    assert not result.ok
    assert result.issues[0].code == "missing_json_file"


def test_missing_split_file_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    _split_path(split_dir, manifest, "val").unlink()
    assert "missing_json_file" in _codes(split_dir)


def test_split_count_mismatch_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["split_counts"]["train"] += 1
    _write_manifest(split_dir, manifest)
    assert "split_count_mismatch" in _codes(split_dir)


def test_total_count_mismatch_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["n_examples"] += 1
    _write_manifest(split_dir, manifest)
    assert "total_count_mismatch" in _codes(split_dir)


def test_label_count_mismatch_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["label_counts"]["1"] += 1
    _write_manifest(split_dir, manifest)
    assert "label_count_mismatch" in _codes(split_dir)


def test_invalid_label_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    payload = _split_payload(split_dir, "train")
    payload["examples"][0]["label"] = 7
    _write_split_payload(split_dir, "train", payload)
    assert "invalid_label" in _codes(split_dir)


def test_phase_flux_length_mismatch_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    payload = _split_payload(split_dir, "train")
    payload["examples"][0]["flux"] = [1.0]
    _write_split_payload(split_dir, "train", payload)
    assert "phase_flux_length_mismatch" in _codes(split_dir)


def test_non_finite_flux_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    payload = _split_payload(split_dir, "train")
    payload["examples"][0]["flux"][0] = float("nan")
    _write_split_payload(split_dir, "train", payload)
    assert "invalid_flux" in _codes(split_dir)


def test_missing_provenance_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    payload = _split_payload(split_dir, "train")
    payload["examples"][0].pop("source_file")
    _write_split_payload(split_dir, "train", payload)
    assert "missing_source_file" in _codes(split_dir)


def test_duplicate_example_id_across_splits_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    train_payload = _split_payload(split_dir, "train")
    val_payload = _split_payload(split_dir, "val")
    val_payload["examples"][0]["example_id"] = train_payload["examples"][0]["example_id"]
    _write_split_payload(split_dir, "val", val_payload)
    assert "duplicate_example_id" in _codes(split_dir)


def test_duplicate_tic_id_across_splits_reports_leakage(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    train_payload = _split_payload(split_dir, "train")
    val_payload = _split_payload(split_dir, "val")
    val_payload["examples"][0]["tic_id"] = train_payload["examples"][0]["tic_id"]
    _write_split_payload(split_dir, "val", val_payload)

    assert "tic_id_leakage" in _codes(split_dir)


def test_live_services_true_reports_error(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["live_services"] = True
    _write_manifest(split_dir, manifest)
    assert "live_services_not_false" in _codes(split_dir)


def test_relative_split_files_are_supported(tmp_path: Path) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["split_files"] = {name: f"{name}.json" for name in ("train", "val", "test")}
    _write_manifest(split_dir, manifest)
    result = validate_split_manifest(split_dir / "manifest.json")
    assert result.ok


def test_cli_returns_nonzero_for_invalid_split(tmp_path: Path, capsys) -> None:
    split_dir = _valid_split_dir(tmp_path)
    manifest = _read_manifest(split_dir)
    manifest["live_services"] = True
    _write_manifest(split_dir, manifest)
    assert _cli([str(split_dir)]) == 1
    assert "Status: FAIL" in capsys.readouterr().out
