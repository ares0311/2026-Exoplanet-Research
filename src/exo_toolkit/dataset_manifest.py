"""Versioned dataset-manifest contract and repository-local validation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DataRole = Literal[
    "training",
    "validation",
    "calibration",
    "frozen_eval",
    "live_search",
    "followup_live_search",
]
CoverageStatus = Literal["known", "not_applicable", "deferred"]
LabelConfidence = Literal["authoritative", "published_review", "mixed", "unlabeled"]


class Coverage(BaseModel):
    """Explicit coverage value, including justified unavailable states."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CoverageStatus
    value: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def require_value_or_reason(self) -> Coverage:
        if self.status == "known" and not self.value:
            raise ValueError("known coverage requires value")
        if self.status != "known" and not self.reason:
            raise ValueError("unavailable coverage requires reason")
        return self


class TargetCoverage(BaseModel):
    """Target namespace and cardinality represented by a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = Field(min_length=1)
    count: int = Field(ge=1)
    selection: str = Field(min_length=1)


class DatasetManifest(BaseModel):
    """Stable, cross-mission dataset provenance contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    project: Literal["2026 Exoplanet Research"]
    role: DataRole
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    target_ids: TargetCoverage
    time_range: Coverage
    cadence: Coverage
    band_or_frequency: str = Field(min_length=1)
    data_product_type: str = Field(min_length=1)
    acquired_at: datetime
    local_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)
    label_source: str = Field(min_length=1)
    label_confidence: LabelConfidence
    preprocessing_version: str = Field(min_length=1)
    known_caveats: tuple[str, ...]
    row_count: int = Field(ge=1)
    group_count: int = Field(ge=1)

    @model_validator(mode="after")
    def group_count_cannot_exceed_rows(self) -> DatasetManifest:
        if self.group_count > self.row_count:
            raise ValueError("group_count cannot exceed row_count")
        return self


class ManifestValidationResult(BaseModel):
    """Validation result suitable for tests, CLI output, and CI."""

    model_config = ConfigDict(frozen=True)

    manifest_path: str
    dataset_id: str | None
    ok: bool
    errors: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Parse and validate one dataset-manifest JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(payload)


def validate_dataset_manifest(
    path: Path,
    *,
    repo_root: Path,
    verify_checksum: bool = True,
) -> ManifestValidationResult:
    """Validate schema, repo-local artifact path, existence, and checksum."""
    errors: list[str] = []
    dataset_id: str | None = None
    try:
        manifest = load_dataset_manifest(path)
        dataset_id = manifest.dataset_id
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ManifestValidationResult(
            manifest_path=str(path), dataset_id=None, ok=False, errors=(str(exc),)
        )

    root = repo_root.resolve()
    artifact = (root / manifest.local_path).resolve()
    try:
        artifact.relative_to(root)
    except ValueError:
        errors.append(f"local_path escapes repository root: {manifest.local_path}")
    else:
        if not artifact.is_file():
            errors.append(f"local artifact is missing: {manifest.local_path}")
        elif verify_checksum:
            observed = sha256_file(artifact)
            if observed != manifest.sha256:
                errors.append(
                    f"sha256 mismatch for {manifest.local_path}: "
                    f"expected {manifest.sha256}, observed {observed}"
                )

    return ManifestValidationResult(
        manifest_path=str(path),
        dataset_id=dataset_id,
        ok=not errors,
        errors=tuple(errors),
    )
