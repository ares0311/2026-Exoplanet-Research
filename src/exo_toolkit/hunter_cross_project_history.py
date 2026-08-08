"""Cross-project Hunter search-history exchange (EXO-Hunter side).

Mirrors ``2026 Technosignatures``'s
``src/techno_search/hunter_cross_project_history.py`` contract exactly --
same ``hunter_prior_search_history_v1`` schema, same
``CROSS_PROJECT_DECISION_STATES``, same per-source/per-entry validity
stamping, same repo-relative ``sibling_history_export_path`` discovery --
so the three Astrometrics Hunter repos agree on one contract rather than
three dialects. Techno-Hunter's module in turn mirrors this repo's own
``hunter_prior_search_history_v1`` shape and
``docs/HUNTER_CROSS_PROJECT_INTERFACE.md``.

Relationship to ``hunter_cross_project.py`` (unchanged, still live)
-------------------------------------------------------------------
That module implements the 2026-07-29 *copy-inward* boundary: sibling
history must be copied inside this repository, is imported into the
append-only ``cross_project_search_history`` table, and is deliberately
never re-read from a sibling checkout. It remains the durable-import path
and is not modified here.

This module adds the *federation read* path the cross-repo directive
requires: a New search may not claim novelty from this project's own
history alone, so it must consult every sibling's published export.
Doing that requires read-only relative-path discovery, which the
2026-07-29 note in ``docs/HUNTER_CROSS_PROJECT_INTERFACE.md`` had
prohibited. That prohibition is superseded for reads only; see that
document's "Superseded" section. Writes into a sibling remain forbidden
(WS-01), as do symlinks, inward copies performed by code, runtime imports
from a sibling, and hardcoded absolute paths (WS-03).

Validity is stamped PER SOURCE and PER ENTRY at load time. There is
deliberately no top-level validity field: an export is only as
trustworthy as its weakest source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CROSS_PROJECT_HISTORY_SCHEMA_VERSION = 1
CROSS_PROJECT_HISTORY_MANIFEST_ID = "hunter-prior-search-history-v1"

#: This project's own key in the federation map.
OWN_PROJECT_KEY = "exo_hunter"

#: Sibling repo directory names, resolved relative to this repo's own
#: location. Never a hardcoded absolute path (WS-03).
CROSS_PROJECT_ROOT_NAMES = {
    "techno_hunter": "2026 Technosignatures",
    "neo_hunter": "2026 Near Earth Objects",
}

#: Repo-relative publish path, identical in all three repos.
CROSS_PROJECT_HISTORY_RELATIVE_PATH = Path("data_selection") / (
    "hunter_prior_search_history_v1.json"
)

#: Operator override for this project's own export location. Changes WHERE
#: the export is read from, never WHETHER it must be decision-grade.
CROSS_PROJECT_HISTORY_PATH_ENV = "EXO_HUNTER_CROSS_PROJECT_HISTORY"

#: Statuses that represent a search that genuinely ran to completion.
#: Shared verbatim with Techno-Hunter so the vocabularies cannot drift.
CROSS_PROJECT_COMPLETED_STATUSES = frozenset(
    {
        "candidate_found",
        "candidate_review_packet",
        "do_not_submit_false_positive",
        "follow_up",
        "human_review_queue",
        "known",
        "known_object_annotation",
        "needs_follow_up_review",
        "no_signal",
        "non_detection",
        "unknown",
        "unresolved",
    }
)

#: Statuses that mean no usable search happened. A target whose only record
#: is ``no_data`` or ``failed`` was NOT searched, so it must not be treated
#: as prior-search evidence.
CROSS_PROJECT_INVALID_STATUSES = frozenset(
    {"cancelled", "failed", "no_data", "not_started"}
)

#: IDENT-03: only these two states may justify a novelty decision.
CROSS_PROJECT_DECISION_STATES = frozenset({"valid", "stale-but-usable"})

#: Ranked weakest-first. Ordering decides only which state gets *reported*
#: as the blocking one; any single non-decision-grade project closes the gate.
HISTORY_STATE_RANK = (
    "invalid",
    "refresh-required",
    "unknown",
    "stale-but-usable",
    "valid",
)

CROSS_PROJECT_HISTORY_DISCLAIMER = (
    "Cross-project Hunter search-history exports are local scheduling aids "
    "shared between independently sandboxed Astrometrics search projects. "
    "They do not constitute a detection, discovery, expert review, external "
    "validation, or authorization for external submission."
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_root() -> Path:
    """This repository's own root, computed from this file's location."""
    return _REPOSITORY_ROOT


def own_history_export_path() -> Path:
    """This repository's own published export path."""
    return _REPOSITORY_ROOT / CROSS_PROJECT_HISTORY_RELATIVE_PATH


def sibling_history_export_path(project: str) -> Path:
    """Resolve a sibling Hunter repo's real, live history-export path.

    Computed relative to this repo's own location, so it reads real,
    current data when the sibling is genuinely checked out as a sibling
    directory -- no symlink, no inward copy, no hardcoded absolute path,
    no runtime import from the sibling (WS-03). Returns the path whether
    or not it exists; absence is handled by the caller as ``unknown``.
    """
    root_name = CROSS_PROJECT_ROOT_NAMES.get(project)
    if root_name is None:
        allowed = ", ".join(sorted(CROSS_PROJECT_ROOT_NAMES))
        raise ValueError(f"unknown sibling project {project!r}; allowed: {allowed}")
    return _REPOSITORY_ROOT.parent / root_name / CROSS_PROJECT_HISTORY_RELATIVE_PATH


def target_alias(raw_id: str) -> str:
    """Normalize a catalog ID to a comparable alias ('TIC 123' -> 'TIC123')."""
    return raw_id.replace(" ", "").upper()


def _parse_timestamp(value: object, *, field: str, path: Path) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Cross-project history {field} is required: {path}")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Cross-project history {field} is invalid: {path}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Cross-project history {field} must be timezone-aware: {path}")
    return parsed.astimezone(UTC)


def _source_validity_state(
    *,
    export_path: Path,
    source_path: str,
    expected_sha256: str,
) -> str:
    """Derive one source's validity by re-hashing its real origin file.

    A direct sibling export sits at ``<repo>/data_selection/<file>``, so the
    origin file resolves against ``<repo>``. An operator-copied export cannot
    reproduce that check (its origin is intentionally absent), so it is
    explicitly ``stale-but-usable`` rather than silently represented as
    current. A present-but-changed origin is ``refresh-required``.
    """
    export_root = (
        export_path.parent.parent if export_path.parent.name == "data_selection" else None
    )
    if export_root is None:
        return "stale-but-usable"
    original = export_root / source_path
    if not original.is_file():
        return "stale-but-usable"
    actual_sha256 = hashlib.sha256(original.read_bytes()).hexdigest()
    return "valid" if actual_sha256 == expected_sha256 else "refresh-required"


def _entry_validity_state(*, status: str, source_validity: str) -> str:
    normalized = status.strip().lower()
    if normalized in CROSS_PROJECT_COMPLETED_STATUSES:
        return source_validity
    if normalized in CROSS_PROJECT_INVALID_STATUSES:
        return "invalid"
    return "unknown"


def load_cross_project_history_export(path: Path) -> dict[str, Any]:
    """Fail-closed structural load of a Hunter history export.

    Stamps ``validity_state`` on every source and every entry. There is no
    top-level validity field by design; callers derive the export's overall
    state from its weakest source.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cross-project history export must be a JSON object: {path}")
    if payload.get("schema_version") != CROSS_PROJECT_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            "Cross-project history export must use schema_version="
            f"{CROSS_PROJECT_HISTORY_SCHEMA_VERSION}: {path}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"Cross-project history export sources must be a non-empty list: {path}"
        )
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Cross-project history export source must be an object: {path}")
        source_project = str(source.get("source_project", "")).strip()
        searched_by = str(source.get("searched_by", "")).strip()
        search_id = str(source.get("search_id", "")).strip()
        started_at = _parse_timestamp(source.get("started_at"), field="started_at", path=path)
        completed_at = _parse_timestamp(
            source.get("completed_at"), field="completed_at", path=path
        )
        source_sha256 = str(source.get("source_sha256", "")).strip().lower()
        source_path = str(source.get("source_path", "")).strip()
        provenance_uri = str(source.get("provenance_uri", "")).strip()
        if (
            not source_project
            or not searched_by
            or not search_id
            or not source_path
            or not provenance_uri
        ):
            raise ValueError(
                f"Cross-project history export source lacks reliable provenance: {path}"
            )
        if completed_at < started_at:
            raise ValueError(
                f"Cross-project history source completed before it started: {path}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError(f"Cross-project history source_sha256 is invalid: {path}")
        source_validity = _source_validity_state(
            export_path=path,
            source_path=source_path,
            expected_sha256=source_sha256,
        )
        if source_validity in {"invalid", "refresh-required"}:
            raise ValueError(
                f"Cross-project history source {source_index} is {source_validity}: {path}"
            )
        source["validity_state"] = source_validity
        entries = source.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Cross-project history export source has no entries: {path}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Cross-project history export entry must be an object: {path}")
            raw_id = str(entry.get("canonical_id") or entry.get("target_id") or "").strip()
            status = str(entry.get("status", "")).strip()
            searched_at = _parse_timestamp(
                entry.get("searched_at"), field="searched_at", path=path
            )
            if not raw_id or not status:
                raise ValueError(
                    "Cross-project history export entry needs target_id/canonical_id "
                    f"and status: {path}"
                )
            if searched_at > completed_at:
                raise ValueError(
                    f"Cross-project history entry occurs after source completion: {path}"
                )
            entry["validity_state"] = _entry_validity_state(
                status=status,
                source_validity=source_validity,
            )
    return payload


def cross_project_alias_counts(payload: Mapping[str, Any]) -> Counter[str]:
    """Count decision-grade cross-project search entries per target alias."""
    counts: Counter[str] = Counter()
    for source in payload.get("sources", []):
        for entry in source.get("entries", []):
            if entry.get("validity_state") not in CROSS_PROJECT_DECISION_STATES:
                continue
            raw_id = str(entry.get("canonical_id") or entry.get("target_id") or "").strip()
            if raw_id:
                counts[target_alias(raw_id)] += 1
    return counts


def cross_project_evidence_by_alias(
    payload: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Map each target alias to its decision-grade cross-project evidence."""
    by_alias: dict[str, list[dict[str, Any]]] = {}
    for source in payload.get("sources", []):
        source_project = str(source.get("source_project") or source.get("searched_by") or "")
        for entry in source.get("entries", []):
            validity_state = str(entry.get("validity_state", "unknown"))
            if validity_state not in CROSS_PROJECT_DECISION_STATES:
                continue
            raw_id = str(entry.get("canonical_id") or entry.get("target_id") or "").strip()
            if not raw_id:
                continue
            by_alias.setdefault(target_alias(raw_id), []).append(
                {
                    "source_project": source_project,
                    "status": str(entry.get("status", "")),
                    "searched_at": str(entry.get("searched_at", "")),
                    "validity_state": validity_state,
                }
            )
    return by_alias


def cross_project_history_validity(
    history_path: Path | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    """Resolve the decision-grade validity of one consumable history export.

    Returns ``(state, detail, payload)``. IDENT-03 permits only ``valid``
    and ``stale-but-usable`` to justify a novelty decision; every other
    state must fail closed. An absent export is ``unknown`` -- never
    silently ``valid``, because absence of a label is not evidence.
    """
    override = os.environ.get(CROSS_PROJECT_HISTORY_PATH_ENV)
    path = history_path or (Path(override) if override else own_history_export_path())
    if not path.is_file():
        return "unknown", f"absent: {path}", None
    try:
        payload = load_cross_project_history_export(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "invalid", f"{path}: {exc}", None
    # Validity is stamped per source; there is no top-level field. The export
    # is only as trustworthy as its weakest source, so a single
    # non-decision-grade source degrades the whole export.
    sources = payload.get("sources") or []
    states = [str(source.get("validity_state", "unknown")) for source in sources]
    if not states:
        return "unknown", f"{path}: no sources", payload
    degraded = [state for state in states if state not in CROSS_PROJECT_DECISION_STATES]
    if degraded:
        return degraded[0], f"{path}: {degraded[0]} source(s)", payload
    state = "stale-but-usable" if "stale-but-usable" in states else "valid"
    return state, f"{path}: {state} across {len(states)} source(s)", payload


def cross_project_history_federation_validity(
    history_path: Path | None = None,
) -> tuple[str, str, dict[str, tuple[str, str]]]:
    """Resolve history validity across THIS project *and* both siblings.

    ``cross_project_history_validity`` answers only "has *this* project
    already searched the target". Novelty is a claim about every
    Astrometrics Hunter project, so a New search must consult all three. A
    sibling that is not checked out, has never published, or published
    something malformed resolves to ``unknown`` -- never silently skipped,
    and never read as evidence of novelty (IDENT-03).

    Sibling exports are read strictly read-only at a path computed relative
    to this repo's own location (WS-03). A sibling that is simply not
    checked out degrades the federation to ``unknown`` instead of raising.

    Returns ``(weakest_state, detail, per_project)``.
    """
    state, detail, per_project, _ = cross_project_history_federation_snapshot(
        history_path
    )
    return state, detail, per_project


def cross_project_history_federation_snapshot(
    history_path: Path | None = None,
) -> tuple[
    str,
    str,
    dict[str, tuple[str, str]],
    dict[str, Any] | None,
]:
    """Validate and freeze the exact federation evidence used for New mode.

    The validity-only API above originally checked live sibling exports and
    then the CLI imported an older repository-local copy.  That split allowed
    identities added after the copy -- including HIP 60759, HIP 61099, and
    HIP 3419 -- to pass the validity gate but remain eligible as New.  This
    function returns one import-compatible snapshot built from the same
    already hash-validated payloads used for the federation decision.

    Only decision-grade entries are included.  Failed/no-data/unrecognized
    records remain in their owning exports but are not converted into prior-
    search evidence.  Per-entry source and export provenance is retained so
    the durable import is auditable without importing sibling code or writing
    outside this repository.
    """
    per_project: dict[str, tuple[str, str]] = {}
    payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    own_path = history_path or (
        Path(override)
        if (override := os.environ.get(CROSS_PROJECT_HISTORY_PATH_ENV))
        else own_history_export_path()
    )
    own_state, own_detail, _ = cross_project_history_validity(history_path)
    per_project[OWN_PROJECT_KEY] = (own_state, own_detail)
    if own_state in CROSS_PROJECT_DECISION_STATES:
        _, _, own_payload = cross_project_history_validity(own_path)
        if own_payload is not None:
            payloads[OWN_PROJECT_KEY] = (own_path, own_payload)

    for project in sorted(CROSS_PROJECT_ROOT_NAMES):
        try:
            sibling_path = sibling_history_export_path(project)
        except ValueError as exc:  # unknown project name -- never assume novelty
            per_project[project] = ("unknown", f"unresolvable sibling: {exc}")
            continue
        # Validated by exactly the same rules as our own export: a sibling is
        # trusted neither more nor less for being remote.
        state, detail, payload = cross_project_history_validity(sibling_path)
        per_project[project] = (state, detail)
        if state in CROSS_PROJECT_DECISION_STATES and payload is not None:
            payloads[project] = (sibling_path, payload)

    weakest = min(
        (state for state, _ in per_project.values()),
        key=lambda state: (
            HISTORY_STATE_RANK.index(state) if state in HISTORY_STATE_RANK else 0
        ),
    )
    detail = "; ".join(
        f"{project}={state} ({project_detail})"
        for project, (state, project_detail) in sorted(per_project.items())
    )
    if weakest not in CROSS_PROJECT_DECISION_STATES:
        return weakest, detail, per_project, None

    merged_sources: list[dict[str, Any]] = []
    for project, (export_path, payload) in sorted(payloads.items()):
        export_sha256 = hashlib.sha256(export_path.read_bytes()).hexdigest()
        for source in payload.get("sources", []):
            entries: list[dict[str, Any]] = []
            for entry in source.get("entries", []):
                if entry.get("validity_state") not in CROSS_PROJECT_DECISION_STATES:
                    continue
                frozen_entry = {
                    key: value
                    for key, value in entry.items()
                    if key != "validity_state"
                }
                frozen_entry["federation_provenance"] = {
                    "project_key": project,
                    "export_path": str(export_path),
                    "export_sha256": export_sha256,
                    "source_path": str(source["source_path"]),
                    "source_sha256": str(source["source_sha256"]),
                    "source_validity_state": str(source["validity_state"]),
                    "entry_validity_state": str(entry["validity_state"]),
                }
                entries.append(frozen_entry)
            merged_sources.append(
                {
                    key: value
                    for key, value in source.items()
                    if key not in {"entries", "validity_state"}
                }
                | {
                    "entries": entries,
                    "federation_project_key": project,
                    "federation_export_path": str(export_path),
                    "federation_export_sha256": export_sha256,
                    "federation_source_validity_state": str(
                        source["validity_state"]
                    ),
                }
            )
    snapshot = {
        "schema_version": CROSS_PROJECT_HISTORY_SCHEMA_VERSION,
        "manifest_id": "hunter-cross-project-federation-snapshot-v1",
        "description": (
            "Read-only, hash-validated snapshot of the current EXO, NEO, and "
            "Techno Hunter history exports used for one New eligibility decision."
        ),
        "sources": merged_sources,
    }
    return weakest, detail, per_project, snapshot


# --------------------------------------------------------------------------
# Publishing this repository's own export
# --------------------------------------------------------------------------

#: EXO-Hunter's own discovery-log directory, relative to the repo root.
DISCOVERY_LOG_GLOB = "discovery_run_*.json"
DISCOVERY_LOG_DIR = Path("logs")
EXO_SOURCE_PROJECT = "2026 Exoplanet Research"
EXO_SEARCHED_BY = "EXO-Hunter"
EXO_METHOD_OR_DATA = (
    "TESS QLP light curves; star_scanner BLS, vetting, and Bayesian scoring"
)


#: ``Skills/star_scanner.py``'s ``ScanLog`` vocabulary -> the shared
#: cross-project vocabulary. This mapping is load-bearing, not cosmetic: an
#: unmapped value stamps as ``unknown`` at load time, which would silently
#: stop a genuinely-searched target from counting as prior-search evidence.
#: ``scanned_clear`` in particular is a real completed search that found
#: nothing, so it must map to a COMPLETED status, while ``error``/``no_data``
#: mean no usable search happened and must map to INVALID ones.
SCANNER_STATUS_TO_CROSS_PROJECT = {
    "candidate_found": "candidate_found",
    "scanned_clear": "no_signal",
    "error": "failed",
    "no_data": "no_data",
}


def _cross_project_status(raw_status: str, *, log_path: Path) -> str:
    """Translate a scanner status, refusing to invent a shared-vocabulary term."""
    mapped = SCANNER_STATUS_TO_CROSS_PROJECT.get(raw_status)
    if mapped is None:
        known = ", ".join(sorted(SCANNER_STATUS_TO_CROSS_PROJECT))
        raise ValueError(
            f"Unmapped scanner status {raw_status!r} in {log_path}; add it to "
            f"SCANNER_STATUS_TO_CROSS_PROJECT (known: {known}). Refusing to "
            "publish a status the shared vocabulary cannot classify."
        )
    return mapped


def _search_id_for(log_path: Path) -> str:
    return "historical-" + log_path.stem.replace("_", "-")


def _mode_for(log_path: Path) -> str:
    """Targeted re-observations of known candidates are follow-up runs."""
    return "follow-up" if "targeted" in log_path.stem else "new"


def _entry_from_scan_record(
    record: Mapping[str, Any], *, log_path: Path
) -> dict[str, Any] | None:
    tic_id = record.get("tic_id")
    searched_at = str(record.get("scanned_at", "")).strip()
    scanner_status = str(record.get("status", "")).strip()
    if tic_id is None or not searched_at or not scanner_status:
        return None
    canonical = f"TIC {int(tic_id)}"
    # ``status`` speaks the shared cross-project vocabulary; the raw scanner
    # term is preserved verbatim under ``result.scanner_status`` so this
    # project's own vocabulary is never lost in translation.
    status = _cross_project_status(scanner_status, log_path=log_path)
    return {
        "target_id": canonical,
        "canonical_id": canonical,
        "mission": "TESS",
        "status": status,
        "searched_at": searched_at,
        "ranking_score": (
            priority_score if (priority_score := record.get("priority_score")) is not None
            else 0.0
        ),
        "best_fpp": record.get("best_fpp"),
        "best_pathway": record.get("best_pathway"),
        "error_message": record.get("error_message"),
        # Fixed allowlist, absent keys dropped rather than emitted as null, so
        # the export stays a summary rather than a copy of the whole scan log
        # (run-007/008 records also carry full per-signal arrays).
        "metrics": {
            key: value
            for key in METRIC_KEYS
            if (value := record.get(key)) is not None
        },
        "result": {
            "best_fpp": record.get("best_fpp"),
            "best_pathway": record.get("best_pathway"),
            "best_period_days": record.get("best_period_days"),
            "scanner_status": scanner_status,
            "signal_count": record.get("n_signals"),
        },
    }


#: Scan-record fields summarized into each entry's ``metrics`` block.
METRIC_KEYS = (
    "n_signals",
    "priority_score",
    "best_period_days",
    "best_fpp",
    "best_pathway",
    "pipeline",
    "exptime",
)


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[int, int, str]:
    """Sort numeric catalog IDs numerically, anything else lexically after."""
    target_id = str(entry.get("target_id", ""))
    _, _, suffix = target_id.partition(" ")
    if suffix.isdigit():
        return (0, int(suffix), target_id)
    return (1, 0, target_id)


def _source_from_discovery_log(log_path: Path, *, repo_root: Path) -> dict[str, Any] | None:
    raw = log_path.read_bytes()
    log = json.loads(raw.decode("utf-8"))
    if not isinstance(log, Mapping):
        raise ValueError(f"Discovery log must be a JSON object: {log_path}")
    # ``ScanLog`` (Skills/star_scanner.py) stores entries as a dict keyed by
    # TIC ID; a plain list is also accepted so a hand-built log still works.
    raw_records = log.get("entries")
    if isinstance(raw_records, Mapping):
        records: Sequence[Any] = list(raw_records.values())
    elif isinstance(raw_records, list):
        records = raw_records
    else:
        return None
    if not records:
        return None
    entries = [
        entry
        for record in records
        if isinstance(record, Mapping)
        and (entry := _entry_from_scan_record(record, log_path=log_path)) is not None
    ]
    if not entries:
        return None
    # Deterministic, catalog-natural order (numeric TIC ascending), matching
    # the committed export so republishing unchanged logs is a no-op.
    entries.sort(key=_entry_sort_key)
    relative = log_path.relative_to(repo_root).as_posix()
    sha256 = hashlib.sha256(raw).hexdigest()
    started_at = min(str(entry["searched_at"]) for entry in entries)
    completed_at = str(log.get("last_updated") or "").strip() or max(
        str(entry["searched_at"]) for entry in entries
    )
    return {
        "search_id": _search_id_for(log_path),
        "mode": _mode_for(log_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "searched_by": EXO_SEARCHED_BY,
        "source_project": EXO_SOURCE_PROJECT,
        "method_or_data": EXO_METHOD_OR_DATA,
        "source_path": relative,
        "source_sha256": sha256,
        "provenance_uri": f"local-artifact:{relative}#sha256={sha256}",
        "entries": entries,
    }


def export_cross_project_history(
    *,
    repo_root: Path | None = None,
    log_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build EXO-Hunter's own portable, schema_version=1 history export.

    Derived from this repo's real ``logs/discovery_run_*.json`` scan logs --
    the same append-only ``ScanLog`` files ``Skills/star_scanner.py`` writes.
    Nothing is invented: ``source_sha256`` is the real hash of the real log
    file, so a consumer re-hashing it resolves ``valid``.
    """
    root = (repo_root or _REPOSITORY_ROOT).resolve()
    paths = (
        sorted(Path(p) for p in log_paths)
        if log_paths is not None
        else sorted((root / DISCOVERY_LOG_DIR).glob(DISCOVERY_LOG_GLOB))
    )
    sources = [
        source
        for path in paths
        if (source := _source_from_discovery_log(path, repo_root=root)) is not None
    ]
    if not sources:
        raise ValueError(
            f"No discovery-run scan logs with entries found under {root / DISCOVERY_LOG_DIR}; "
            "nothing real to export"
        )
    return {
        "schema_version": CROSS_PROJECT_HISTORY_SCHEMA_VERSION,
        "manifest_id": CROSS_PROJECT_HISTORY_MANIFEST_ID,
        "description": (
            "Normalized append-only EXO-Hunter (2026 Exoplanet Research) real "
            "production search history; source discovery logs remain unchanged. "
            "Status values are this project's own scanner vocabulary, not a "
            "shared outcome classification."
        ),
        "disclaimer": CROSS_PROJECT_HISTORY_DISCLAIMER,
        "sources": sources,
    }


def write_cross_project_history_export(
    output_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    log_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Write EXO-Hunter's own portable history export and return a summary."""
    root = (repo_root or _REPOSITORY_ROOT).resolve()
    target = output_path or (root / CROSS_PROJECT_HISTORY_RELATIVE_PATH)
    payload = export_cross_project_history(repo_root=root, log_paths=log_paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entry_count = sum(len(source["entries"]) for source in payload["sources"])
    unique_targets = {
        entry["target_id"] for source in payload["sources"] for entry in source["entries"]
    }
    return {
        "schema_version": CROSS_PROJECT_HISTORY_SCHEMA_VERSION,
        "ok": True,
        "disclaimer": CROSS_PROJECT_HISTORY_DISCLAIMER,
        "output_path": str(target),
        "source_count": len(payload["sources"]),
        "entry_count": entry_count,
        "unique_target_count": len(unique_targets),
    }
