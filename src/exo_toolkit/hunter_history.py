"""Fail-closed loading and source verification for Hunter history manifests."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _repository_root_for(manifest_path: Path) -> Path:
    resolved = manifest_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return resolved.parent


def resolve_history_source_path(
    manifest: Path | Mapping[str, Any],
    source_path: str,
    *,
    source_root: Path | None = None,
) -> Path:
    """Resolve one history source exactly as the fail-closed loader does."""
    if isinstance(manifest, Path):
        root = source_root.resolve() if source_root else _repository_root_for(manifest)
    else:
        root = (source_root or Path.cwd()).resolve()
    candidate = Path(source_path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def load_verified_history_manifest(
    manifest: Path | Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Load a history manifest and verify every declared source byte-for-byte."""
    if isinstance(manifest, Path):
        manifest_path = manifest.resolve()
        raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("History manifest must be a JSON object")
        payload: dict[str, Any] = raw_payload
        resolved_root = source_root.resolve() if source_root else _repository_root_for(
            manifest_path
        )
    else:
        payload = dict(manifest)
        resolved_root = (source_root or Path.cwd()).resolve()

    if payload.get("schema_version") != 1:
        raise ValueError("History manifest must use schema_version=1")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("History manifest sources must be a non-empty list")

    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("History manifest sources must be objects")
        source_path_text = str(source.get("source_path", "")).strip()
        if not source_path_text:
            raise ValueError("History source is missing provenance field source_path")
        expected = str(source.get("source_sha256", ""))
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(
                f"History source {source.get('search_id', '<unknown>')} needs a lowercase SHA-256"
            )
        resolved_path = resolve_history_source_path(
            manifest,
            source_path_text,
            source_root=resolved_root,
        )
        if not resolved_path.is_file():
            raise ValueError(
                f"History source file does not exist: {source_path_text} "
                f"(resolved to {resolved_path})"
            )
        actual = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"History source hash mismatch for {source_path_text}: "
                f"expected={expected} actual={actual}"
            )
    return payload
