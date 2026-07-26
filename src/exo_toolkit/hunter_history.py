"""Fail-closed loading and source verification for Hunter history manifests.

Both the file-manifest and in-memory dict-manifest branches share the same
repo-root walk-up heuristic (``_walk_up_for_repo_root``) when
``source_root`` is not given explicitly. The dict-manifest branch
previously fell back to a bare ``Path.cwd()`` with no attempt to locate
the real repo root -- dormant against every current call site (both
``star_scanner.py`` and ``batch_scan.py`` already pass ``source_root``
explicitly) but the same latent trap already demonstrated as exploitable
for the file-manifest case (a manifest resolved from inside an unrelated
nested repo silently uses that repo's root).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def _walk_up_for_repo_root(start_dir: Path) -> Path:
    """Walk upward from ``start_dir`` (inclusive) for the repo-root marker
    (``pyproject.toml`` + ``src/``); fall back to ``start_dir`` itself."""
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return start_dir


def _repository_root_for(manifest_path: Path) -> Path:
    return _walk_up_for_repo_root(manifest_path.resolve().parent)


def build_manual_scan_source(
    *,
    script: str,
    log_path: Path,
    entries: Sequence[Mapping[str, Any]],
    started_at: datetime,
    completed_at: datetime,
    method_or_data: str,
    mode: str = "new",
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Build one history-manifest "source" for a completed manual CLI scan.

    Uses exactly the schema ``HunterStore.import_history_manifest`` already
    verifies for the seven curated legacy-log imports (byte-for-byte source
    hash, non-empty typed entries), so a standalone ``star_scanner.py`` or
    ``batch_scan.py`` CLI run is durably recorded in Hunter's
    ``target_search_history`` through the exact same fail-closed code path
    instead of being a silent bypass of the durable pipeline. The returned
    dict is meant to be wrapped as ``{"schema_version": 1, "sources": [...]}"``
    and passed straight to ``import_history_manifest`` -- no manifest file
    needs to be written to disk.

    ``search_id`` is derived from the scan log file's own content hash, so
    re-running this on an unchanged log is naturally idempotent (an unchanged
    ``entries`` payload reproduces the same computed identity hash inside
    ``import_history_manifest`` and is skipped as already-imported); a scan
    log with newly appended entries produces a new, distinct search_id
    representing a fresh, additional search event.
    """
    resolved_log = log_path.resolve()
    if not resolved_log.is_file():
        raise ValueError(f"Manual scan log does not exist: {resolved_log}")
    if not entries:
        raise ValueError("Manual scan source requires at least one entry")
    source_sha256 = hashlib.sha256(resolved_log.read_bytes()).hexdigest()
    root = (source_root or _repository_root_for(resolved_log)).resolve()
    try:
        source_path = str(resolved_log.relative_to(root))
    except ValueError:
        source_path = str(resolved_log)
    search_id = f"manual-{script}-{source_sha256[:24]}"
    return {
        "search_id": search_id,
        "mode": mode,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "searched_by": f"manual_cli:{script}",
        "source_project": "2026 Exoplanet Research",
        "method_or_data": method_or_data,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "provenance_uri": f"manual-scan:{script}:{search_id}",
        "entries": [dict(entry) for entry in entries],
    }


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
        root = source_root.resolve() if source_root else _walk_up_for_repo_root(Path.cwd())
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
        resolved_root = (
            source_root.resolve() if source_root else _walk_up_for_repo_root(Path.cwd())
        )

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
