"""Stable read-only cross-project Hunter history interchange."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CROSS_PROJECT_SCHEMA_VERSION = 1
SUPPORTED_SIBLINGS = {"technosignatures": "2026 Technosignatures"}
DEFAULT_COPIED_HISTORY = (
    Path(__file__).resolve().parents[2]
    / "data_selection"
    / "cross_project_imports"
    / "techno_hunter_history_v1.json"
)


def sibling_history_export_path(project: str) -> Path:
    """Resolve a sibling's export relative to this checkout, never absolutely."""
    directory = SUPPORTED_SIBLINGS.get(project.lower())
    if directory is None:
        raise ValueError(
            f"unknown sibling {project!r}; allowed: {', '.join(sorted(SUPPORTED_SIBLINGS))}"
        )
    return (
        Path(__file__).resolve().parents[3]
        / directory
        / "data_selection"
        / "hunter_prior_search_history_v1.json"
    )


def sibling_project_root(project: str) -> Path:
    """Return the computed sibling root for independent source-hash checks."""
    return sibling_history_export_path(project).parents[1]


def sibling_history_source_path(project: str) -> Path:
    """Return the sibling's verified raw history when no normalized export exists."""
    if project.lower() != "technosignatures":
        raise ValueError(f"no raw-history adapter exists for sibling {project!r}")
    return sibling_project_root(project) / "results" / "scan_history.ndjson"


def build_sibling_history_manifest(project: str) -> dict[str, Any]:
    """Normalize a supported sibling's real read-only history into schema v1."""
    source = sibling_history_source_path(project)
    if not source.is_file():
        raise ValueError(f"sibling history source does not exist: {source}")
    source_bytes = source.read_bytes()
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source_bytes.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid sibling history JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, Mapping) or row.get("schema_version") != (
            "prod_scan_history_v1"
        ):
            raise ValueError(
                f"unsupported sibling history schema at {source}:{line_number}"
            )
        target_stem = str(row.get("target_stem", ""))
        match = re.search(
            r"(?i)(?:^|[_-])HIP[_ -]?(\d+)(?:[_\-.]|$)",
            target_stem,
        )
        if match is None:
            continue
        searched_at = str(row.get("scanned_at_utc", "")).strip()
        if not searched_at:
            raise ValueError(
                f"sibling history row lacks scanned_at_utc at {source}:{line_number}"
            )
        identity = f"HIP {int(match.group(1))}"
        entries.append(
            {
                "target_id": identity,
                "canonical_id": identity,
                "aliases": [identity],
                "searched_at": searched_at,
                "status": str(row.get("pathway") or "searched"),
                "source_record": dict(row),
            }
        )
    if not entries:
        raise ValueError(f"sibling history contains no interoperable identities: {source}")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    return {
        "schema_version": CROSS_PROJECT_SCHEMA_VERSION,
        "manifest_id": f"technosignatures-live-{source_sha256[:16]}",
        "sources": [
            {
                "search_id": f"techno-prod-scan-history-{source_sha256[:16]}",
                "source_project": SUPPORTED_SIBLINGS[project.lower()],
                "source_path": "results/scan_history.ndjson",
                "source_sha256": source_sha256,
                "entries": entries,
            }
        ],
    }


def normalize_stellar_identity(value: object) -> str | None:
    """Normalize interoperable TIC/HIP/KIC identities."""
    text = " ".join(str(value or "").strip().upper().replace("_", " ").split())
    compact = text.replace(" ", "")
    for prefix in ("TIC", "HIP", "KIC"):
        suffix = compact.removeprefix(prefix)
        if compact.startswith(prefix) and suffix.isdigit():
            return f"{prefix} {int(suffix)}"
    return None


def load_cross_project_history(
    path: Path | Mapping[str, Any],
    *,
    source_root: Path | None,
) -> dict[str, Any]:
    """Load and independently verify a sibling/copy history manifest when possible."""
    if isinstance(path, Mapping):
        payload = dict(path)
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        manifest_path = f"inline:{payload.get('manifest_id', 'cross-project-history')}"
    else:
        if not path.is_file():
            raise ValueError(f"cross-project history export does not exist: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
        manifest_path = str(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("cross-project history must be a schema_version=1 object")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("cross-project history requires a non-empty sources list")

    source_hashes_verified = 0
    normalized_entries: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources, 1):
        if not isinstance(source, Mapping):
            raise ValueError(f"cross-project source {source_index} must be an object")
        source_project = str(source.get("source_project", "")).strip()
        source_path = str(source.get("source_path", "")).strip()
        source_sha256 = str(source.get("source_sha256", "")).strip()
        if not source_project or not source_path:
            raise ValueError(f"cross-project source {source_index} lacks provenance")
        if len(source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in source_sha256
        ):
            raise ValueError(f"cross-project source {source_index} has invalid SHA-256")
        if source_root is not None:
            resolved = (source_root / source_path).resolve()
            try:
                resolved.relative_to(source_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"cross-project source escapes source root: {source_path}"
                ) from exc
            if not resolved.is_file():
                raise ValueError(f"cross-project source artifact is missing: {resolved}")
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if actual != source_sha256:
                raise ValueError(
                    f"cross-project source hash mismatch: {resolved}; "
                    f"expected={source_sha256} actual={actual}"
                )
            source_hashes_verified += 1
        entries = source.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"cross-project source {source_index} entries must be a list")
        for entry_index, entry in enumerate(entries, 1):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"cross-project source {source_index} entry {entry_index} "
                    "must be an object"
                )
            identities = {
                identity
                for raw_identity in (
                    entry.get("target_id"),
                    entry.get("canonical_id"),
                    *(entry.get("aliases") or ()),
                )
                if (identity := normalize_stellar_identity(raw_identity)) is not None
            }
            if not identities:
                continue
            searched_at = str(entry.get("searched_at", "")).strip()
            status = str(entry.get("status", "")).strip()
            if not searched_at or not status:
                raise ValueError(
                    f"cross-project source {source_index} entry {entry_index} "
                    "lacks searched_at/status"
                )
            normalized_entries.append(
                {
                    "source_project": source_project,
                    "source_search_id": str(source.get("search_id", "")),
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "identities": sorted(identities),
                    "searched_at": searched_at,
                    "status": status,
                    "source_entry": dict(entry),
                }
            )
    if not normalized_entries:
        raise ValueError("cross-project history contains no interoperable stellar identities")
    return {
        "schema_version": CROSS_PROJECT_SCHEMA_VERSION,
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "validity_state": (
            "valid" if source_hashes_verified == len(sources) else "stale-but-usable"
        ),
        "source_hashes_verified": source_hashes_verified,
        "source_count": len(sources),
        "entries": normalized_entries,
    }
