from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo_toolkit.dataset_manifest import (
    DatasetManifest,
    load_dataset_manifest,
    sha256_file,
    validate_dataset_manifest,
)


def _payload(artifact: Path, *, digest: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "test_dataset_v1",
        "project": "2026 Exoplanet Research",
        "role": "frozen_eval",
        "source_name": "Test source",
        "source_url": "https://example.invalid/source",
        "instrument": "Test instrument",
        "target_ids": {"namespace": "TEST", "count": 2, "selection": "fixture"},
        "time_range": {"status": "not_applicable", "reason": "catalog only"},
        "cadence": {"status": "known", "value": "2 minutes"},
        "band_or_frequency": "optical",
        "data_product_type": "fixture",
        "acquired_at": "2026-07-10T00:00:00Z",
        "local_path": artifact.as_posix(),
        "sha256": digest,
        "license": "test fixture",
        "label_source": "fixture labels",
        "label_confidence": "published_review",
        "preprocessing_version": "test-v1",
        "known_caveats": [],
        "row_count": 2,
        "group_count": 2,
    }


def test_load_and_validate_manifest_checksum(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_payload(Path("artifact.jsonl"), digest=sha256_file(artifact))),
        encoding="utf-8",
    )

    manifest = load_dataset_manifest(manifest_path)
    result = validate_dataset_manifest(manifest_path, repo_root=tmp_path)

    assert manifest.dataset_id == "test_dataset_v1"
    assert result.ok
    assert result.errors == ()


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_payload(Path("artifact.jsonl"), digest="0" * 64)),
        encoding="utf-8",
    )

    result = validate_dataset_manifest(manifest_path, repo_root=tmp_path)

    assert not result.ok
    assert "sha256 mismatch" in result.errors[0]


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_payload(Path("../outside.jsonl"), digest="0" * 64)),
        encoding="utf-8",
    )

    result = validate_dataset_manifest(manifest_path, repo_root=tmp_path)

    assert not result.ok
    assert "escapes repository root" in result.errors[0]


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    payload = _payload(Path("artifact.jsonl"), digest="0" * 64)
    payload["surprise"] = True

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(payload)


def test_known_coverage_requires_value(tmp_path: Path) -> None:
    payload = _payload(Path("artifact.jsonl"), digest="0" * 64)
    payload["cadence"] = {"status": "known"}

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(payload)


def test_group_count_cannot_exceed_rows(tmp_path: Path) -> None:
    payload = _payload(Path("artifact.jsonl"), digest="0" * 64)
    payload["group_count"] = 3

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(payload)


def test_all_committed_dataset_manifests_validate() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = sorted((repo_root / "metadata/dataset_manifests").glob("*.json"))

    assert paths
    for path in paths:
        result = validate_dataset_manifest(path, repo_root=repo_root)
        assert result.ok, result.errors


def test_committed_json_schema_has_contract_required_fields() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    schema = json.loads(
        (repo_root / "metadata/dataset_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(DatasetManifest.model_fields) == set(schema["required"])
