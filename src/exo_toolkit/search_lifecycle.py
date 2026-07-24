"""Durable Hunter search lifecycle shared by the EXO-Hunter shell commands."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from exo_toolkit.hunter_history import (
    load_verified_history_manifest,
    resolve_history_source_path,
)
from exo_toolkit.hunter_models import (
    EXECUTABLE_SEARCH_STATES,
    TERMINAL_TARGET_STATUSES,
    ExecutionProvenance,
    FollowUpRecommendation,
    HunterCandidate,
    PriorSearch,
    SearchExecutionSummary,
    SearchMode,
    TargetExecutionResult,
)
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_CONFIDENCE_MIN,
    FOLLOW_UP_FPP_MAX,
    FOLLOW_UP_PATHWAYS,
    FOLLOW_UP_SELECTOR_VERSION,
)

HUNTER_SCHEMA_VERSION = 5


class _ClosingConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


_DDL = """
CREATE TABLE IF NOT EXISTS candidate_catalog (
    snapshot_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidate_catalog_target
    ON candidate_catalog(target_id, observed_at);

CREATE TABLE IF NOT EXISTS search_manifests (
    search_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('new', 'follow-up')),
    requested_target_count INTEGER NOT NULL CHECK(requested_target_count > 0),
    selected_target_count INTEGER NOT NULL CHECK(selected_target_count > 0),
    candidate_pool_count INTEGER NOT NULL CHECK(candidate_pool_count >= selected_target_count),
    selector_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_manifest_targets (
    search_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    ranking_score REAL NOT NULL,
    selection_reason TEXT NOT NULL,
    PRIMARY KEY(search_id, ordinal),
    UNIQUE(search_id, target_id),
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id),
    FOREIGN KEY(snapshot_id) REFERENCES candidate_catalog(snapshot_id)
);

CREATE TABLE IF NOT EXISTS search_state_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'pending', 'running', 'interrupted', 'partial', 'failed',
        'completed', 'archived_partial'
    )),
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS search_runs (
    attempt_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'running', 'interrupted', 'partial', 'failed', 'completed'
    )),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    config_json TEXT NOT NULL,
    items_processed INTEGER NOT NULL DEFAULT 0 CHECK(items_processed >= 0),
    items_succeeded INTEGER NOT NULL DEFAULT 0 CHECK(items_succeeded >= 0),
    items_failed INTEGER NOT NULL DEFAULT 0 CHECK(items_failed >= 0),
    error_message TEXT,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS target_search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'candidate_found', 'no_signal', 'no_data', 'failed'
    )),
    result_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id),
    FOREIGN KEY(attempt_id) REFERENCES search_runs(attempt_id)
);
CREATE INDEX IF NOT EXISTS idx_target_history_target
    ON target_search_history(target_id, id);

CREATE TABLE IF NOT EXISTS follow_up_registry (
    follow_up_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    priority REAL NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    prior_search_provenance_json TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    search_eligible INTEGER NOT NULL DEFAULT 1 CHECK(search_eligible IN (0, 1)),
    revisit_reason TEXT,
    parent_follow_up_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('open', 'scheduled', 'completed', 'deferred')),
    created_at TEXT NOT NULL,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS follow_up_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    follow_up_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'open', 'scheduled', 'attempt_failed', 'completed', 'deferred'
    )),
    related_search_id TEXT,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(follow_up_id) REFERENCES follow_up_registry(follow_up_id),
    FOREIGN KEY(related_search_id) REFERENCES search_manifests(search_id)
);
CREATE INDEX IF NOT EXISTS idx_follow_up_events_follow_up
    ON follow_up_events(follow_up_id, id);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS hunter_immutable_candidate_catalog_update
BEFORE UPDATE ON candidate_catalog BEGIN
    SELECT RAISE(ABORT, 'candidate_catalog is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_candidate_catalog_delete
BEFORE DELETE ON candidate_catalog BEGIN
    SELECT RAISE(ABORT, 'candidate_catalog is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_search_manifests_update
BEFORE UPDATE ON search_manifests BEGIN
    SELECT RAISE(ABORT, 'search_manifests is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_search_manifests_delete
BEFORE DELETE ON search_manifests BEGIN
    SELECT RAISE(ABORT, 'search_manifests is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_manifest_targets_update
BEFORE UPDATE ON search_manifest_targets BEGIN
    SELECT RAISE(ABORT, 'search_manifest_targets is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_manifest_targets_delete
BEFORE DELETE ON search_manifest_targets BEGIN
    SELECT RAISE(ABORT, 'search_manifest_targets is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_search_state_update
BEFORE UPDATE ON search_state_events BEGIN
    SELECT RAISE(ABORT, 'search_state_events is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_search_state_delete
BEFORE DELETE ON search_state_events BEGIN
    SELECT RAISE(ABORT, 'search_state_events is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_target_history_update
BEFORE UPDATE ON target_search_history BEGIN
    SELECT RAISE(ABORT, 'target_search_history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_target_history_delete
BEFORE DELETE ON target_search_history BEGIN
    SELECT RAISE(ABORT, 'target_search_history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_follow_up_events_update
BEFORE UPDATE ON follow_up_events BEGIN
    SELECT RAISE(ABORT, 'follow_up_events is append-only');
END;
CREATE TRIGGER IF NOT EXISTS hunter_immutable_follow_up_events_delete
BEFORE DELETE ON follow_up_events BEGIN
    SELECT RAISE(ABORT, 'follow_up_events is append-only');
END;
"""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def format_eta(seconds: float) -> str:
    """Format a nonnegative ETA without impossible 60-second remainders."""
    if math.isinf(seconds):
        return "unknown"
    rounded = max(0, int(round(seconds)))
    if rounded > 90:
        minutes, remainder = divmod(rounded, 60)
        return f"{minutes}m{remainder:02d}s"
    return f"{rounded}s"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, payload: Any, now: datetime) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:12]
    return f"{prefix}-{now.strftime('%Y%m%dT%H%M%SZ')}-{digest}"


class HunterStore:
    """SQLite system of record for exact, resumable Hunter searches."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(_DDL)
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(follow_up_registry)")
            }
            if "search_eligible" not in columns:
                connection.execute(
                    "ALTER TABLE follow_up_registry "
                    "ADD COLUMN search_eligible INTEGER NOT NULL DEFAULT 1"
                )
            if "revisit_reason" not in columns:
                connection.execute(
                    "ALTER TABLE follow_up_registry ADD COLUMN revisit_reason TEXT"
                )
            if "parent_follow_up_id" not in columns:
                connection.execute(
                    "ALTER TABLE follow_up_registry ADD COLUMN parent_follow_up_id TEXT"
                )
            existing_events = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT follow_up_id FROM follow_up_events"
                ).fetchall()
            }
            for row in connection.execute(
                "SELECT follow_up_id, status, created_at FROM follow_up_registry"
            ).fetchall():
                follow_up_id = str(row["follow_up_id"])
                if follow_up_id in existing_events:
                    continue
                created_at = datetime.fromisoformat(str(row["created_at"]))
                self._append_follow_up_event(
                    connection,
                    follow_up_id=follow_up_id,
                    state=str(row["status"]),
                    related_search_id=None,
                    detail={"migration_backfill": True},
                    created_at=created_at,
                )
            migration_at = _utc_now()
            deferred_rows = connection.execute(
                "SELECT follow_up_id, search_id, revisit_reason "
                "FROM follow_up_registry "
                "WHERE status='open' AND search_eligible=0"
            ).fetchall()
            for row in deferred_rows:
                if row["revisit_reason"] is None:
                    raise RuntimeError(
                        "Non-executable follow-up is missing its required revisit reason: "
                        f"{row['follow_up_id']}"
                    )
                connection.execute(
                    "UPDATE follow_up_registry SET status='deferred' "
                    "WHERE follow_up_id=?",
                    (row["follow_up_id"],),
                )
                self._append_follow_up_event(
                    connection,
                    follow_up_id=str(row["follow_up_id"]),
                    state="deferred",
                    related_search_id=str(row["search_id"]),
                    detail={
                        "previous_state": "open",
                        "migration_correction": (
                            "non-executable follow-up with an explicit revisit gate"
                        ),
                    },
                    created_at=migration_at,
                )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations "
                "(version, applied_at, description) VALUES (?, ?, ?)",
                (
                    HUNTER_SCHEMA_VERSION,
                    _utc_now().isoformat(),
                    "Storage-enforced immutable Hunter history and validity contract",
                ),
            )
            connection.execute(f"PRAGMA user_version = {HUNTER_SCHEMA_VERSION}")

    def import_reviewed_follow_up(
        self,
        *,
        candidate: HunterCandidate | Mapping[str, Any],
        recommendation: FollowUpRecommendation | Mapping[str, Any],
        source_search_id: str,
        source_attempt_id: str,
        source_result: Mapping[str, Any],
        source_provenance: Mapping[str, Any],
        completed_at: datetime,
        imported_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Import one provenance-complete prior result and its explicit revisit policy."""
        validated_candidate = (
            candidate
            if isinstance(candidate, HunterCandidate)
            else HunterCandidate.model_validate(candidate)
        )
        validated_recommendation = (
            recommendation
            if isinstance(recommendation, FollowUpRecommendation)
            else FollowUpRecommendation.model_validate(recommendation)
        )
        if validated_candidate.source_provenance.get("search_category") != "follow-up":
            raise ValueError("reviewed import candidate must have search_category=follow-up")
        if not source_search_id or not source_attempt_id:
            raise ValueError("source search and attempt IDs are required")
        if completed_at.tzinfo is None:
            raise ValueError("completed_at must include a timezone")

        identity_payload = {
            "source_search_id": source_search_id,
            "source_attempt_id": source_attempt_id,
            "candidate": validated_candidate.model_dump(mode="json"),
            "result": dict(source_result),
            "provenance": dict(source_provenance),
            "recommendation": validated_recommendation.model_dump(mode="json"),
        }
        search_id = _stable_id("prior-search", identity_payload, completed_at)
        attempt_id = _stable_id(
            "prior-attempt",
            {"search_id": search_id, "source_attempt_id": source_attempt_id},
            completed_at,
        )
        created_at = imported_at or _utc_now()
        manifest_sha256 = hashlib.sha256(_canonical_json(identity_payload).encode()).hexdigest()
        candidate_json = validated_candidate.model_dump(mode="json")
        snapshot_id = hashlib.sha256(
            f"{search_id}:1:{_canonical_json(candidate_json)}".encode()
        ).hexdigest()

        with self.connect() as connection:
            existing = connection.execute(
                "SELECT manifest_sha256 FROM search_manifests WHERE search_id=?",
                (search_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["manifest_sha256"]) != manifest_sha256:
                    raise RuntimeError(f"Imported search identity collision: {search_id}")
                follow_up = connection.execute(
                    "SELECT follow_up_id FROM follow_up_registry WHERE search_id=?",
                    (search_id,),
                ).fetchone()
                if follow_up is None:
                    raise RuntimeError(f"Imported search is missing its follow-up row: {search_id}")
                return {
                    "created": False,
                    "search_id": search_id,
                    "attempt_id": attempt_id,
                    "follow_up_id": str(follow_up["follow_up_id"]),
                }

            config = {
                "imported_prior_search": True,
                "source_search_id": source_search_id,
                "source_attempt_id": source_attempt_id,
            }
            connection.execute(
                """
                INSERT INTO search_manifests (
                    search_id, schema_version, mode, requested_target_count,
                    selected_target_count, candidate_pool_count, selector_version,
                    config_json, manifest_sha256, created_at
                ) VALUES (?, ?, 'follow-up', 1, 1, 1, ?, ?, ?, ?)
                """,
                (
                    search_id,
                    HUNTER_SCHEMA_VERSION,
                    "reviewed_prior_import_v1",
                    _canonical_json(config),
                    manifest_sha256,
                    completed_at.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO candidate_catalog "
                "(snapshot_id, search_id, target_id, canonical_id, candidate_json, observed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    search_id,
                    validated_candidate.target_id,
                    validated_candidate.canonical_id,
                    _canonical_json(candidate_json),
                    created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO search_manifest_targets (
                    search_id, ordinal, snapshot_id, target_id, canonical_id,
                    ranking_score, selection_reason
                ) VALUES (?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    search_id,
                    snapshot_id,
                    validated_candidate.target_id,
                    validated_candidate.canonical_id,
                    validated_candidate.ranking_score,
                    validated_candidate.selection_reason,
                ),
            )
            connection.execute(
                """
                INSERT INTO search_runs (
                    attempt_id, search_id, status, started_at, completed_at,
                    config_json, items_processed, items_succeeded, items_failed
                ) VALUES (?, ?, 'completed', ?, ?, ?, 1, 1, 0)
                """,
                (
                    attempt_id,
                    search_id,
                    completed_at.isoformat(),
                    completed_at.isoformat(),
                    _canonical_json(config),
                ),
            )
            connection.execute(
                """
                INSERT INTO target_search_history (
                    search_id, attempt_id, target_id, status, result_json,
                    provenance_json, error_message, created_at
                ) VALUES (?, ?, ?, 'candidate_found', ?, ?, NULL, ?)
                """,
                (
                    search_id,
                    attempt_id,
                    validated_candidate.target_id,
                    _canonical_json(dict(source_result)),
                    _canonical_json(dict(source_provenance)),
                    completed_at.isoformat(),
                ),
            )
            self._append_state(
                connection,
                search_id,
                "completed",
                {
                    "imported_prior_search": True,
                    "source_search_id": source_search_id,
                    "source_attempt_id": source_attempt_id,
                },
                created_at,
            )
            if not self._register_follow_up(
                connection,
                search_id=search_id,
                candidate=validated_candidate,
                follow_up=validated_recommendation,
                created_at=created_at,
            ):
                raise RuntimeError("Reviewed follow-up import did not create a registry row")
            follow_up = connection.execute(
                "SELECT follow_up_id FROM follow_up_registry WHERE search_id=?",
                (search_id,),
            ).fetchone()
        return {
            "created": True,
            "search_id": search_id,
            "attempt_id": attempt_id,
            "follow_up_id": str(follow_up["follow_up_id"]),
        }

    def import_history_manifest(
        self,
        manifest: Path | Mapping[str, Any],
        *,
        source_root: Path | None = None,
    ) -> dict[str, int]:
        """Idempotently import provenance-complete historical project searches."""
        payload = load_verified_history_manifest(
            manifest,
            source_root=source_root,
        )
        sources = payload.get("sources")
        assert isinstance(sources, list)  # verified by load_verified_history_manifest

        sources_created = 0
        events_created = 0
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for source in sources:
                if not isinstance(source, Mapping):
                    raise ValueError("History manifest sources must be objects")
                search_id = str(source.get("search_id", ""))
                mode = str(source.get("mode", ""))
                started_at = datetime.fromisoformat(str(source.get("started_at", "")))
                completed_at = datetime.fromisoformat(str(source.get("completed_at", "")))
                entries = source.get("entries")
                if not search_id or mode not in {"new", "follow-up"}:
                    raise ValueError("Each history source requires search_id and valid mode")
                if started_at.tzinfo is None or completed_at.tzinfo is None:
                    raise ValueError(
                        f"History source {search_id} timestamps need timezone"
                    )
                source_hash = str(source.get("source_sha256", ""))
                if len(source_hash) != 64 or any(
                    character not in "0123456789abcdef" for character in source_hash
                ):
                    raise ValueError(
                        f"History source {search_id} needs a lowercase SHA-256"
                    )
                for field in (
                    "searched_by",
                    "source_project",
                    "method_or_data",
                    "source_path",
                    "provenance_uri",
                ):
                    if not str(source.get(field, "")).strip():
                        raise ValueError(
                            f"History source {search_id} is missing provenance field {field}"
                        )
                if not isinstance(entries, list) or not entries:
                    raise ValueError(f"History source {search_id} entries must be non-empty")
                source_payload = dict(source)
                source_sha256 = hashlib.sha256(
                    _canonical_json(source_payload).encode()
                ).hexdigest()
                existing = connection.execute(
                    "SELECT manifest_sha256 FROM search_manifests WHERE search_id=?",
                    (search_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["manifest_sha256"]) != source_sha256:
                        raise RuntimeError(
                            f"Historical search identity collision: {search_id}"
                        )
                    continue

                normalized: list[tuple[HunterCandidate, dict[str, Any]]] = []
                seen_targets: set[str] = set()
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        raise ValueError(f"History source {search_id} has a non-object entry")
                    target_id = str(entry.get("target_id", ""))
                    if not target_id or target_id in seen_targets:
                        raise ValueError(
                            f"History source {search_id} has missing/duplicate target_id"
                        )
                    seen_targets.add(target_id)
                    status = str(entry.get("status", ""))
                    if status not in {"candidate_found", "no_signal", "no_data", "failed"}:
                        raise ValueError(
                            f"History source {search_id} has invalid status={status!r}"
                        )
                    searched_at = datetime.fromisoformat(str(entry.get("searched_at", "")))
                    if searched_at.tzinfo is None:
                        raise ValueError(
                            f"History source {search_id} entry timestamp needs timezone"
                        )
                    if status == "failed" and not entry.get("error_message"):
                        raise ValueError(
                            f"History source {search_id} failed entry needs error_message"
                        )
                    mission = entry.get("mission", "TESS")
                    if mission not in {"TESS", "Kepler", "K2", "JWST"}:
                        raise ValueError(
                            f"History source {search_id} has invalid mission={mission!r}"
                        )
                    ranking_score = float(entry.get("ranking_score", 0.0))
                    candidate = HunterCandidate(
                        target_id=target_id,
                        canonical_id=str(entry.get("canonical_id", target_id)),
                        aliases=tuple(str(value) for value in entry.get("aliases", ())),
                        mission=mission,
                        source="Imported historical project search",
                        source_provenance={
                            "search_category": "historical",
                            "historical_search_id": search_id,
                            "source_path": source.get("source_path"),
                            "source_sha256": source.get("source_sha256"),
                        },
                        eligible=False,
                        eligibility_reason="historical search snapshot",
                        ranking_score=ranking_score,
                        selection_reason="Preserved exact historical search membership",
                        metrics=dict(entry.get("metrics", {})),
                    )
                    normalized.append((candidate, dict(entry)))

                config = {
                    "historical_import": True,
                    "source_path": source.get("source_path"),
                    "source_sha256": source.get("source_sha256"),
                    "source_payload": source_payload,
                    "source_artifact_path": str(
                        resolve_history_source_path(
                            manifest,
                            str(source.get("source_path")),
                            source_root=source_root,
                        )
                    ),
                    "method_or_data": source.get("method_or_data"),
                }
                connection.execute(
                    """
                    INSERT INTO search_manifests (
                        search_id, schema_version, mode, requested_target_count,
                        selected_target_count, candidate_pool_count, selector_version,
                        config_json, manifest_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_id,
                        HUNTER_SCHEMA_VERSION,
                        mode,
                        len(normalized),
                        len(normalized),
                        len(normalized),
                        "historical_project_import_v1",
                        _canonical_json(config),
                        source_sha256,
                        completed_at.isoformat(),
                    ),
                )
                attempt_id = f"historical-attempt-{source_sha256[:24]}"
                failed = sum(entry[1]["status"] == "failed" for entry in normalized)
                succeeded = len(normalized) - failed
                run_status = "completed" if failed == 0 else "partial"
                lifecycle_state = "completed" if failed == 0 else "archived_partial"
                connection.execute(
                    """
                    INSERT INTO search_runs (
                        attempt_id, search_id, status, started_at, completed_at,
                        config_json, items_processed, items_succeeded, items_failed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        search_id,
                        run_status,
                        started_at.isoformat(),
                        completed_at.isoformat(),
                        _canonical_json(config),
                        len(normalized),
                        succeeded,
                        failed,
                    ),
                )
                for ordinal, (candidate, entry) in enumerate(normalized, 1):
                    candidate_json = candidate.model_dump(mode="json")
                    snapshot_id = hashlib.sha256(
                        f"{search_id}:{ordinal}:{_canonical_json(candidate_json)}".encode()
                    ).hexdigest()
                    connection.execute(
                        "INSERT INTO candidate_catalog "
                        "(snapshot_id, search_id, target_id, canonical_id, "
                        "candidate_json, observed_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            snapshot_id,
                            search_id,
                            candidate.target_id,
                            candidate.canonical_id,
                            _canonical_json(candidate_json),
                            completed_at.isoformat(),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO search_manifest_targets (
                            search_id, ordinal, snapshot_id, target_id, canonical_id,
                            ranking_score, selection_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            search_id,
                            ordinal,
                            snapshot_id,
                            candidate.target_id,
                            candidate.canonical_id,
                            candidate.ranking_score,
                            candidate.selection_reason,
                        ),
                    )
                    result = dict(entry.get("result", {}))
                    result["historical_entry"] = entry
                    provenance = {
                        "searched_by": str(source.get("searched_by", "EXO-Hunter")),
                        "source_project": str(
                            source.get("source_project", "2026 Exoplanet Research")
                        ),
                        "method_or_data": str(source.get("method_or_data", "unknown")),
                        "provenance_uri": str(source.get("provenance_uri", "unknown")),
                        "source_path": source.get("source_path"),
                        "source_sha256": source.get("source_sha256"),
                    }
                    connection.execute(
                        """
                        INSERT INTO target_search_history (
                            search_id, attempt_id, target_id, status, result_json,
                            provenance_json, error_message, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            search_id,
                            attempt_id,
                            candidate.target_id,
                            entry["status"],
                            _canonical_json(result),
                            _canonical_json(provenance),
                            entry.get("error_message"),
                            str(entry.get("searched_at", completed_at.isoformat())),
                        ),
                    )
                self._append_state(
                    connection,
                    search_id,
                    lifecycle_state,
                    {
                        "historical_import": True,
                        "items_processed": len(normalized),
                        "items_failed": failed,
                    },
                    completed_at,
                )
                sources_created += 1
                events_created += len(normalized)
        return {
            "sources_total": len(sources),
            "sources_created": sources_created,
            "events_created": events_created,
        }

    def create_search(
        self,
        candidates: Sequence[HunterCandidate | Mapping[str, Any]],
        *,
        requested_target_count: int,
        mode: SearchMode,
        selector_version: str,
        config: Mapping[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Freeze a ranked universe and exact pending manifest atomically."""
        if requested_target_count < 1:
            raise ValueError("requested_target_count must be at least 1")
        current_open = self.open_searches()
        if current_open:
            raise RuntimeError(
                "An executable search already exists; run or resolve it before creating "
                f"another: {', '.join(row['search_id'] for row in current_open)}"
            )

        validated = [
            value if isinstance(value, HunterCandidate) else HunterCandidate.model_validate(value)
            for value in candidates
        ]
        if len({row.target_id for row in validated}) != len(validated):
            raise ValueError("candidate universe contains duplicate target_id values")

        if mode == "new":
            searched = self.searched_target_ids()
            eligible = [
                row for row in validated if row.eligible and row.target_id not in searched
            ]
        else:
            eligible = [row for row in validated if row.eligible and row.prior_searches]

        eligible.sort(key=lambda row: (-row.ranking_score, row.canonical_id, row.target_id))
        if not eligible:
            raise RuntimeError(
                f"No {mode} candidates are currently available from "
                f"{len(validated)} evaluated candidates; no search was created"
            )
        selected = eligible[:requested_target_count]
        created_at = now or _utc_now()
        identity_payload = {
            "mode": mode,
            "requested": requested_target_count,
            "targets": [row.model_dump(mode="json") for row in selected],
            "config": dict(config),
            "selector_version": selector_version,
        }
        search_id = _stable_id("exo-search", identity_payload, created_at)
        manifest_sha256 = hashlib.sha256(_canonical_json(identity_payload).encode()).hexdigest()

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            concurrent_open = connection.execute(
                """
                SELECT m.search_id, e.state
                FROM search_manifests AS m
                JOIN search_state_events AS e ON e.id = (
                    SELECT MAX(e2.id) FROM search_state_events AS e2
                    WHERE e2.search_id = m.search_id
                )
                WHERE e.state IN ('pending', 'running', 'partial', 'failed')
                ORDER BY m.created_at
                """
            ).fetchall()
            if concurrent_open:
                raise RuntimeError(
                    "An executable search was created concurrently; run or resolve it "
                    "before creating another: "
                    + ", ".join(str(row["search_id"]) for row in concurrent_open)
                )
            connection.execute(
                """
                INSERT INTO search_manifests (
                    search_id, schema_version, mode, requested_target_count,
                    selected_target_count, candidate_pool_count, selector_version,
                    config_json, manifest_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    search_id,
                    HUNTER_SCHEMA_VERSION,
                    mode,
                    requested_target_count,
                    len(selected),
                    len(validated),
                    selector_version,
                    _canonical_json(dict(config)),
                    manifest_sha256,
                    created_at.isoformat(),
                ),
            )
            snapshot_ids: dict[str, str] = {}
            for index, candidate in enumerate(validated, 1):
                candidate_json = candidate.model_dump(mode="json")
                snapshot_id = hashlib.sha256(
                    f"{search_id}:{index}:{_canonical_json(candidate_json)}".encode()
                ).hexdigest()
                snapshot_ids[candidate.target_id] = snapshot_id
                connection.execute(
                    "INSERT INTO candidate_catalog "
                    "(snapshot_id, search_id, target_id, canonical_id, "
                    "candidate_json, observed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot_id,
                        search_id,
                        candidate.target_id,
                        candidate.canonical_id,
                        _canonical_json(candidate_json),
                        created_at.isoformat(),
                    ),
                )
            for ordinal, candidate in enumerate(selected, 1):
                connection.execute(
                    """
                    INSERT INTO search_manifest_targets (
                        search_id, ordinal, snapshot_id, target_id, canonical_id,
                        ranking_score, selection_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_id,
                        ordinal,
                        snapshot_ids[candidate.target_id],
                        candidate.target_id,
                        candidate.canonical_id,
                        candidate.ranking_score,
                        candidate.selection_reason,
                    ),
                )
            self._append_state(
                connection,
                search_id,
                "pending",
                {"selected_target_count": len(selected)},
                created_at,
            )
            if mode == "follow-up":
                for candidate in selected:
                    follow_up_id = candidate.source_provenance.get("follow_up_id")
                    if follow_up_id is None:
                        continue
                    self._transition_follow_up(
                        connection,
                        follow_up_id=str(follow_up_id),
                        expected_state="open",
                        new_state="scheduled",
                        related_search_id=search_id,
                        detail={"target_id": candidate.target_id},
                        created_at=created_at,
                    )
        return self.get_search(search_id)

    @staticmethod
    def _append_state(
        connection: sqlite3.Connection,
        search_id: str,
        state: str,
        detail: Mapping[str, Any],
        created_at: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO search_state_events "
            "(search_id, state, detail_json, created_at) VALUES (?, ?, ?, ?)",
            (search_id, state, _canonical_json(dict(detail)), created_at.isoformat()),
        )

    @staticmethod
    def _append_follow_up_event(
        connection: sqlite3.Connection,
        *,
        follow_up_id: str,
        state: str,
        related_search_id: str | None,
        detail: Mapping[str, Any],
        created_at: datetime,
    ) -> None:
        payload = {
            "follow_up_id": follow_up_id,
            "state": state,
            "related_search_id": related_search_id,
            "detail": dict(detail),
            "created_at": created_at.isoformat(),
        }
        event_id = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        connection.execute(
            "INSERT INTO follow_up_events "
            "(event_id, follow_up_id, state, related_search_id, detail_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event_id,
                follow_up_id,
                state,
                related_search_id,
                _canonical_json(dict(detail)),
                created_at.isoformat(),
            ),
        )

    @classmethod
    def _transition_follow_up(
        cls,
        connection: sqlite3.Connection,
        *,
        follow_up_id: str,
        expected_state: str,
        new_state: str,
        related_search_id: str,
        detail: Mapping[str, Any],
        created_at: datetime,
        search_eligible: bool | None = None,
        revisit_reason: str | None = None,
    ) -> None:
        row = connection.execute(
            "SELECT status, search_eligible, revisit_reason "
            "FROM follow_up_registry WHERE follow_up_id=?",
            (follow_up_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Unknown follow-up registry row: {follow_up_id}")
        current_state = str(row["status"])
        if current_state != expected_state:
            raise RuntimeError(
                f"Follow-up {follow_up_id} is not {expected_state}: state={current_state}"
            )
        next_eligible = (
            int(bool(row["search_eligible"]))
            if search_eligible is None
            else int(search_eligible)
        )
        next_revisit_reason = (
            str(row["revisit_reason"])
            if revisit_reason is None and row["revisit_reason"] is not None
            else revisit_reason
        )
        connection.execute(
            "UPDATE follow_up_registry SET status=?, search_eligible=?, revisit_reason=? "
            "WHERE follow_up_id=?",
            (new_state, next_eligible, next_revisit_reason, follow_up_id),
        )
        cls._append_follow_up_event(
            connection,
            follow_up_id=follow_up_id,
            state=new_state,
            related_search_id=related_search_id,
            detail={
                "previous_state": expected_state,
                "previous_revisit_reason": row["revisit_reason"],
                **dict(detail),
            },
            created_at=created_at,
        )

    def current_state(self, search_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM search_state_events WHERE search_id=? ORDER BY id DESC LIMIT 1",
                (search_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown search_id: {search_id}")
        return str(row["state"])

    def open_searches(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*, e.state
                FROM search_manifests AS m
                JOIN search_state_events AS e ON e.id = (
                    SELECT MAX(e2.id) FROM search_state_events AS e2
                    WHERE e2.search_id = m.search_id
                )
                ORDER BY m.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows if row["state"] in EXECUTABLE_SEARCH_STATES]

    def searched_target_ids(self) -> frozenset[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT target_id FROM target_search_history "
                "WHERE status IN ('candidate_found', 'no_signal', 'no_data')"
            ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def get_search(self, search_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            manifest = connection.execute(
                "SELECT * FROM search_manifests WHERE search_id=?", (search_id,)
            ).fetchone()
            if manifest is None:
                raise KeyError(f"Unknown search_id: {search_id}")
            targets = connection.execute(
                """
                SELECT mt.*, cc.candidate_json
                FROM search_manifest_targets AS mt
                JOIN candidate_catalog AS cc ON cc.snapshot_id = mt.snapshot_id
                WHERE mt.search_id=? ORDER BY mt.ordinal
                """,
                (search_id,),
            ).fetchall()
            state = connection.execute(
                "SELECT state FROM search_state_events WHERE search_id=? ORDER BY id DESC LIMIT 1",
                (search_id,),
            ).fetchone()
        result = dict(manifest)
        result["config"] = json.loads(result.pop("config_json"))
        result["state"] = str(state["state"])
        result["targets"] = [
            {**dict(row), "candidate": json.loads(row["candidate_json"])} for row in targets
        ]
        for row in result["targets"]:
            row.pop("candidate_json", None)
        return result

    def execute_search(
        self,
        runner: Callable[[HunterCandidate], TargetExecutionResult | Mapping[str, Any]],
        *,
        search_id: str | None = None,
        workers: int = 6,
        run_config: Mapping[str, Any] | None = None,
        now_fn: Callable[[], datetime] = _utc_now,
        progress_fn: Callable[[str], None] | None = None,
    ) -> SearchExecutionSummary:
        """Execute only the frozen manifest; retry only non-terminal targets."""
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if search_id is None:
            open_rows = self.open_searches()
            if not open_rows:
                raise RuntimeError("No pending or resumable search exists")
            if len(open_rows) > 1:
                raise RuntimeError("Multiple executable searches exist; pass search_id explicitly")
            search_id = str(open_rows[0]["search_id"])
        search = self.get_search(search_id)
        if search["state"] not in EXECUTABLE_SEARCH_STATES:
            raise RuntimeError(f"Search {search_id} is not executable: state={search['state']}")

        completed_ids = self.completed_target_ids(search_id)
        pending = [
            HunterCandidate.model_validate(row["candidate"])
            for row in search["targets"]
            if row["target_id"] not in completed_ids
        ]
        if not pending:
            raise RuntimeError(f"Search {search_id} has no remaining targets")

        started_at = now_fn()
        attempt_payload = {"search_id": search_id, "started_at": started_at.isoformat()}
        attempt_id = _stable_id("attempt", attempt_payload, started_at)
        run_config_dict = dict(run_config or {})
        with self.connect() as connection:
            interrupted = connection.execute(
                """
                UPDATE search_runs
                SET status='interrupted', completed_at=?,
                    error_message='Previous process ended before finalizing this attempt'
                WHERE search_id=? AND status='running'
                """,
                (started_at.isoformat(), search_id),
            ).rowcount
            if interrupted:
                self._append_state(
                    connection,
                    search_id,
                    "interrupted",
                    {"recovered_attempt_count": interrupted},
                    started_at,
                )
            connection.execute(
                "INSERT INTO search_runs "
                "(attempt_id, search_id, status, started_at, config_json) "
                "VALUES (?, ?, 'running', ?, ?)",
                (
                    attempt_id,
                    search_id,
                    started_at.isoformat(),
                    _canonical_json(run_config_dict),
                ),
            )
            self._append_state(
                connection,
                search_id,
                "running",
                {"attempt_id": attempt_id, "remaining_targets": len(pending)},
                started_at,
            )

        start_monotonic = time.monotonic()
        succeeded = 0
        failed = 0
        follow_up_count = 0

        def _safe_run(candidate: HunterCandidate) -> TargetExecutionResult:
            try:
                raw = runner(candidate)
                outcome = (
                    raw
                    if isinstance(raw, TargetExecutionResult)
                    else TargetExecutionResult.model_validate(raw)
                )
                if isinstance(outcome.provenance, ExecutionProvenance):
                    return outcome
                normalized = ExecutionProvenance(
                    candidate_snapshot=candidate,
                    pipeline_context={
                        "run_config": run_config_dict,
                        "runner_provenance": outcome.provenance,
                    },
                    code_version=str(run_config_dict.get("code_version", "unknown")),
                    git_commit=str(run_config_dict.get("git_commit", "unknown")),
                    scorer=str(run_config_dict.get("scorer", "unknown")),
                    model_artifacts=tuple(run_config_dict.get("model_artifacts", ())),
                    runner=getattr(runner, "__name__", type(runner).__name__),
                    failure_stage="runner" if outcome.status == "failed" else None,
                )
                return outcome.model_copy(update={"provenance": normalized})
            except Exception as exc:  # noqa: BLE001
                return TargetExecutionResult(
                    status="failed",
                    result={},
                    provenance=ExecutionProvenance(
                        candidate_snapshot=candidate,
                        pipeline_context={"run_config": run_config_dict},
                        code_version=str(run_config_dict.get("code_version", "unknown")),
                        git_commit=str(run_config_dict.get("git_commit", "unknown")),
                        scorer=str(run_config_dict.get("scorer", "unknown")),
                        model_artifacts=tuple(run_config_dict.get("model_artifacts", ())),
                        runner=getattr(runner, "__name__", type(runner).__name__),
                        failure_stage="runner",
                    ),
                    error_message=f"{type(exc).__name__}: {exc}",
                )

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_safe_run, candidate): candidate for candidate in pending}
                for done, future in enumerate(as_completed(futures), 1):
                    candidate = futures[future]
                    outcome = future.result()
                    assert isinstance(outcome.provenance, ExecutionProvenance)
                    created_at = now_fn()
                    with self.connect() as connection:
                        connection.execute(
                            """
                            INSERT INTO target_search_history (
                                search_id, attempt_id, target_id, status, result_json,
                                provenance_json, error_message, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                search_id,
                                attempt_id,
                                candidate.target_id,
                                outcome.status,
                                _canonical_json(outcome.result),
                                _canonical_json(outcome.provenance.model_dump(mode="json")),
                                outcome.error_message,
                                created_at.isoformat(),
                            ),
                        )
                        for follow_up in outcome.follow_ups:
                            if self._register_follow_up(
                                connection,
                                search_id=search_id,
                                candidate=candidate,
                                follow_up=follow_up,
                                created_at=created_at,
                            ):
                                follow_up_count += 1
                        source_follow_up_id = candidate.source_provenance.get("follow_up_id")
                        if source_follow_up_id is not None and outcome.status == "failed":
                            self._append_follow_up_event(
                                connection,
                                follow_up_id=str(source_follow_up_id),
                                state="attempt_failed",
                                related_search_id=search_id,
                                detail={
                                    "attempt_id": attempt_id,
                                    "target_id": candidate.target_id,
                                    "error_message": outcome.error_message,
                                },
                                created_at=created_at,
                            )
                        elif (
                            source_follow_up_id is not None
                            and outcome.status in TERMINAL_TARGET_STATUSES
                        ):
                            disposition = (
                                "deferred" if outcome.status == "no_data" else "completed"
                            )
                            self._transition_follow_up(
                                connection,
                                follow_up_id=str(source_follow_up_id),
                                expected_state="scheduled",
                                new_state=disposition,
                                related_search_id=search_id,
                                detail={
                                    "attempt_id": attempt_id,
                                    "target_id": candidate.target_id,
                                    "outcome_status": outcome.status,
                                    "error_message": outcome.error_message,
                                },
                                created_at=created_at,
                                search_eligible=False,
                                revisit_reason=(
                                    f"Follow-up search {search_id} returned no data; "
                                    "new archive products are required before another search."
                                    if outcome.status == "no_data"
                                    else None
                                ),
                            )
                    if outcome.status in TERMINAL_TARGET_STATUSES:
                        succeeded += 1
                    else:
                        failed += 1
                    if progress_fn is not None:
                        elapsed = time.monotonic() - start_monotonic
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remaining = (len(pending) - done) / rate if rate > 0 else float("inf")
                        progress_fn(
                            f"[{done}/{len(pending)}] target={candidate.target_id} "
                            f"status={outcome.status} elapsed={elapsed:.0f}s "
                            f"ETA={format_eta(remaining)}"
                        )
        except BaseException as exc:
            completed_at = now_fn()
            self._finalize_attempt(
                attempt_id,
                search_id,
                status="failed",
                processed=succeeded + failed,
                succeeded=succeeded,
                failed=failed,
                completed_at=completed_at,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            raise

        completed_at = now_fn()
        all_complete = len(self.completed_target_ids(search_id)) == int(
            search["selected_target_count"]
        )
        final_status: Literal["completed", "partial", "failed"]
        if all_complete:
            final_status = "completed"
        elif succeeded == 0 and not completed_ids:
            final_status = "failed"
        else:
            final_status = "partial"
        self._finalize_attempt(
            attempt_id,
            search_id,
            status=final_status,
            processed=succeeded + failed,
            succeeded=succeeded,
            failed=failed,
            completed_at=completed_at,
            error_message=None,
        )
        return SearchExecutionSummary(
            search_id=search_id,
            attempt_id=attempt_id,
            status=final_status,
            started_at=started_at,
            completed_at=completed_at,
            targets_total=int(search["selected_target_count"]),
            targets_already_complete=len(completed_ids),
            targets_processed=succeeded + failed,
            targets_succeeded=succeeded,
            targets_failed=failed,
            follow_ups_registered=follow_up_count,
        )

    def _finalize_attempt(
        self,
        attempt_id: str,
        search_id: str,
        *,
        status: Literal["completed", "partial", "failed"],
        processed: int,
        succeeded: int,
        failed: int,
        completed_at: datetime,
        error_message: str | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE search_runs SET status=?, completed_at=?, items_processed=?,
                    items_succeeded=?, items_failed=?, error_message=?
                WHERE attempt_id=? AND status='running'
                """,
                (
                    status,
                    completed_at.isoformat(),
                    processed,
                    succeeded,
                    failed,
                    error_message,
                    attempt_id,
                ),
            )
            self._append_state(
                connection,
                search_id,
                status,
                {
                    "attempt_id": attempt_id,
                    "items_processed": processed,
                    "items_succeeded": succeeded,
                    "items_failed": failed,
                    "error_message": error_message,
                },
                completed_at,
            )

    @classmethod
    def _register_follow_up(
        cls,
        connection: sqlite3.Connection,
        *,
        search_id: str,
        candidate: HunterCandidate,
        follow_up: FollowUpRecommendation,
        created_at: datetime,
    ) -> bool:
        parent_follow_up_id = candidate.source_provenance.get("follow_up_id")
        if parent_follow_up_id is not None:
            parent = connection.execute(
                "SELECT 1 FROM follow_up_registry WHERE follow_up_id=?",
                (str(parent_follow_up_id),),
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Unknown parent follow-up: {parent_follow_up_id}")
        follow_up_id = hashlib.sha256(
            f"{search_id}:{candidate.target_id}:{follow_up.candidate_id}".encode()
        ).hexdigest()
        initial_state = "open" if follow_up.search_eligible else "deferred"
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO follow_up_registry (
                follow_up_id, search_id, target_id, candidate_id, priority, reason,
                evidence_json, prior_search_provenance_json, recommended_action,
                search_eligible, revisit_reason, parent_follow_up_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                follow_up_id,
                search_id,
                candidate.target_id,
                follow_up.candidate_id,
                follow_up.priority,
                follow_up.reason,
                _canonical_json(follow_up.evidence),
                _canonical_json(
                    [row.model_dump(mode="json") for row in candidate.prior_searches]
                    + [
                        {
                            "searched_by": "EXO-Hunter",
                            "searched_at": created_at.isoformat(),
                            "source_project": "2026 Exoplanet Research",
                            "method_or_data": "durable EXO-Hunter lifecycle execution",
                            "result": follow_up.reason,
                            "provenance_uri": f"hunter-search:{search_id}",
                        }
                    ]
                ),
                follow_up.recommended_action,
                int(follow_up.search_eligible),
                follow_up.revisit_reason,
                parent_follow_up_id,
                initial_state,
                created_at.isoformat(),
            ),
        )
        created = cursor.rowcount == 1
        if created:
            cls._append_follow_up_event(
                connection,
                follow_up_id=follow_up_id,
                state=initial_state,
                related_search_id=search_id,
                detail={
                    "target_id": candidate.target_id,
                    "candidate_id": follow_up.candidate_id,
                    "parent_follow_up_id": parent_follow_up_id,
                    "search_eligible": follow_up.search_eligible,
                },
                created_at=created_at,
            )
        return created

    def completed_target_ids(self, search_id: str) -> frozenset[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT target_id FROM target_search_history "
                "WHERE search_id=? AND status IN ('candidate_found', 'no_signal', 'no_data')",
                (search_id,),
            ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def target_history(self, target_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM target_search_history WHERE target_id=? ORDER BY id",
                (target_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["result"] = json.loads(row.pop("result_json"))
            row["provenance"] = json.loads(row.pop("provenance_json"))
        return result

    def list_follow_ups(self, *, status: str = "open") -> list[dict[str, Any]]:
        with self.connect() as connection:
            if status == "all":
                rows = connection.execute(
                    "SELECT * FROM follow_up_registry "
                    "ORDER BY priority DESC, created_at, follow_up_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM follow_up_registry WHERE status=? "
                    "ORDER BY priority DESC, created_at, follow_up_id",
                    (status,),
                ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["evidence"] = json.loads(row.pop("evidence_json"))
            row["prior_search_provenance"] = json.loads(
                row.pop("prior_search_provenance_json")
            )
            row["search_eligible"] = bool(row["search_eligible"])
            row["events"] = self.follow_up_events(str(row["follow_up_id"]))
        return result

    def follow_up_events(self, follow_up_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM follow_up_events WHERE follow_up_id=? "
                "ORDER BY id",
                (follow_up_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["detail"] = json.loads(row.pop("detail_json"))
        return result

    @staticmethod
    def _history_result_summary(
        row: Mapping[str, Any],
    ) -> tuple[str, float | None, float | None, str | None]:
        result = json.loads(str(row["result_json"]))
        historical = result.get("historical_entry", {})
        composite = result.get("composite_result", {})
        scores = composite.get("scores", {}) if isinstance(composite, Mapping) else {}
        fpp = historical.get("best_fpp") if isinstance(historical, Mapping) else None
        if fpp is None and isinstance(composite, Mapping):
            fpp = composite.get(
                "false_positive_probability",
                scores.get("false_positive_probability"),
            )
        confidence = (
            historical.get("best_detection_confidence")
            if isinstance(historical, Mapping)
            else None
        )
        if confidence is None and isinstance(composite, Mapping):
            confidence = composite.get(
                "detection_confidence",
                scores.get("detection_confidence"),
            )
        pathway = historical.get("best_pathway") if isinstance(historical, Mapping) else None
        if pathway is None and isinstance(composite, Mapping):
            pathway = composite.get("pathway")
        numeric_fpp = float(fpp) if isinstance(fpp, (int, float)) else None
        numeric_confidence = (
            float(confidence) if isinstance(confidence, (int, float)) else None
        )
        summary = str(row["status"])
        if numeric_fpp is not None:
            summary += f"; FPP={numeric_fpp:.6f}"
        if pathway:
            summary += f"; pathway={pathway}"
        return (
            summary,
            numeric_fpp,
            numeric_confidence,
            str(pathway) if pathway else None,
        )

    def follow_up_universe(self) -> list[HunterCandidate]:
        """Build the deterministic follow-up universe from every durable history."""
        with self.connect() as connection:
            history_rows = connection.execute(
                """
                SELECT h.*, m.mode, c.candidate_json
                FROM target_search_history AS h
                JOIN search_manifests AS m ON m.search_id=h.search_id
                LEFT JOIN candidate_catalog AS c
                    ON c.search_id=h.search_id AND c.target_id=h.target_id
                ORDER BY h.target_id, h.id
                """
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in history_rows:
            grouped.setdefault(str(raw["target_id"]), []).append(dict(raw))

        registry_by_target: dict[str, list[dict[str, Any]]] = {}
        for row in self.list_follow_ups(status="all"):
            registry_by_target.setdefault(str(row["target_id"]), []).append(row)

        candidates: list[HunterCandidate] = []
        for target_id in sorted(set(grouped) | set(registry_by_target)):
            histories = grouped.get(target_id, [])
            registry_rows = registry_by_target.get(target_id, [])
            prior_searches: list[PriorSearch] = []
            seen_prior: set[tuple[str, str]] = set()
            template: HunterCandidate | None = None
            for row in histories:
                candidate_json = row.get("candidate_json")
                if candidate_json:
                    parsed = HunterCandidate.model_validate(json.loads(str(candidate_json)))
                    template = parsed
                    for prior in parsed.prior_searches:
                        key = (prior.provenance_uri, prior.searched_at.isoformat())
                        if key not in seen_prior:
                            prior_searches.append(prior)
                            seen_prior.add(key)
                provenance = json.loads(str(row["provenance_json"]))
                summary, _, _, _ = self._history_result_summary(row)
                prior = PriorSearch(
                    searched_by=str(provenance.get("searched_by", "EXO-Hunter")),
                    searched_at=datetime.fromisoformat(str(row["created_at"])),
                    source_project=str(
                        provenance.get("source_project", "2026 Exoplanet Research")
                    ),
                    method_or_data=str(
                        provenance.get(
                            "method_or_data",
                            "EXO-Hunter acquisition, preprocessing, vetting, and scoring",
                        )
                    ),
                    result=summary,
                    provenance_uri=str(
                        provenance.get(
                            "provenance_uri",
                            f"hunter-search:{row['search_id']}/{row['attempt_id']}",
                        )
                    ),
                )
                key = (prior.provenance_uri, prior.searched_at.isoformat())
                if key not in seen_prior:
                    prior_searches.append(prior)
                    seen_prior.add(key)
            if not prior_searches:
                raise RuntimeError(f"Follow-up target lacks durable prior history: {target_id}")

            if registry_rows:
                registry_rows.sort(
                    key=lambda row: (
                        row["status"] != "open",
                        -float(row["priority"]),
                        str(row["follow_up_id"]),
                    )
                )
                registry = registry_rows[0]
                # Availability, not scientific quality: a registry row already
                # encodes true non-availability (already scheduled/completed, or
                # explicitly revisit-gated for missing data) as a non-"open"
                # status. FPP/confidence/pathway are reported separately as the
                # strict production bar; they never remove a real signal from
                # the ranked follow-up pool (see selection_contract()).
                eligible = bool(registry["status"] == "open")
                eligibility_reason = (
                    "actionable evidence-based registry recommendation"
                    if eligible
                    else f"registry disposition={registry['status']}"
                )
                ranking_score = float(registry["priority"])
                selection_reason = str(registry["reason"])
                source_provenance = {
                    "search_category": "follow-up",
                    "selector_version": FOLLOW_UP_SELECTOR_VERSION,
                    "follow_up_id": registry["follow_up_id"],
                    "search_id": registry["search_id"],
                    "history_event_count": len(histories),
                }
                metrics: dict[str, float | int | str | None] = {
                    "follow_up_priority": ranking_score,
                    "expected_information_gain": ranking_score / 110.0,
                    "scientific_suitability": ranking_score / 110.0,
                    "prior_search_count": len(prior_searches),
                    "registry_status": str(registry["status"]),
                    "meets_strict_follow_up_bar": registry["evidence"].get(
                        "meets_strict_production_bar"
                    ),
                }
            else:
                latest = histories[-1]
                summary, fpp, confidence, pathway = self._history_result_summary(latest)
                # Availability, not scientific quality: any unfollowed candidate_found
                # signal is a legitimate follow-up candidate. FPP/confidence/pathway
                # inform ranking_score and the reported strict-bar flag below; they no
                # longer remove real signals from the selectable pool (directive: rank
                # and absolute quality are different).
                eligible = bool(latest["mode"] == "new" and latest["status"] == "candidate_found")
                meets_strict_bar = bool(
                    fpp is not None
                    and fpp < FOLLOW_UP_FPP_MAX
                    and confidence is not None
                    and confidence > FOLLOW_UP_CONFIDENCE_MIN
                    and pathway in FOLLOW_UP_PATHWAYS
                )
                if latest["mode"] == "follow-up":
                    eligibility_reason = "latest durable search already performed follow-up"
                elif latest["status"] != "candidate_found":
                    eligibility_reason = f"latest result={latest['status']}"
                elif meets_strict_bar:
                    eligibility_reason = "new-search evidence passes strict follow-up bar"
                else:
                    eligibility_reason = (
                        "new-search evidence found a signal below the strict follow-up bar"
                    )
                ranking_score = (
                    100.0 * (1.0 - fpp) + 10.0 * confidence
                    if fpp is not None and confidence is not None
                    else 0.0
                )
                selection_reason = f"Durable prior-search evaluation: {summary}"
                source_provenance = {
                    "search_category": "follow-up",
                    "selector_version": FOLLOW_UP_SELECTOR_VERSION,
                    "search_id": latest["search_id"],
                    "history_event_count": len(histories),
                }
                metrics = {
                    "prior_search_count": len(prior_searches),
                    "latest_status": str(latest["status"]),
                    "latest_mode": str(latest["mode"]),
                    "latest_fpp": fpp,
                    "latest_detection_confidence": confidence,
                    "latest_pathway": pathway,
                    "expected_information_gain": (
                        (1.0 - fpp) * confidence
                        if fpp is not None and confidence is not None
                        else None
                    ),
                    "scientific_suitability": confidence,
                    "meets_strict_follow_up_bar": meets_strict_bar if eligible else None,
                }

            candidates.append(
                HunterCandidate(
                    target_id=target_id,
                    canonical_id=(template.canonical_id if template else target_id),
                    aliases=(template.aliases if template else ()),
                    mission=(template.mission if template else "TESS"),
                    object_classification=(
                        template.object_classification if template else "star"
                    ),
                    source="Durable EXO-Hunter prior-search universe",
                    source_provenance=source_provenance,
                    eligible=eligible,
                    eligibility_reason=eligibility_reason,
                    distance_pc=(template.distance_pc if template else None),
                    estimated_download_gb=(
                        template.estimated_download_gb if template else None
                    ),
                    ranking_score=ranking_score,
                    selection_reason=selection_reason,
                    metrics=metrics,
                    prior_searches=tuple(prior_searches),
                )
            )
        candidates.sort(key=lambda row: (-row.ranking_score, row.canonical_id, row.target_id))
        return candidates

    def follow_up_candidates(self) -> list[HunterCandidate]:
        """Return only currently executable rows from the full history universe."""
        return [candidate for candidate in self.follow_up_universe() if candidate.eligible]

    def export_manifest_csv(self, search_id: str, path: Path) -> Path:
        search = self.get_search(search_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "ordinal",
            "target_id",
            "canonical_id",
            "mission",
            "object_classification",
            "distance_pc",
            "estimated_download_gb",
            "ranking_score",
            "selection_reason",
            "source",
            "prior_search_provenance",
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for target in search["targets"]:
                candidate = HunterCandidate.model_validate(target["candidate"])
                writer.writerow(
                    {
                        "ordinal": target["ordinal"],
                        "target_id": candidate.target_id,
                        "canonical_id": candidate.canonical_id,
                        "mission": candidate.mission,
                        "object_classification": candidate.object_classification,
                        "distance_pc": candidate.distance_pc,
                        "estimated_download_gb": candidate.estimated_download_gb,
                        "ranking_score": candidate.ranking_score,
                        "selection_reason": candidate.selection_reason,
                        "source": candidate.source,
                        "prior_search_provenance": _canonical_json(
                            [row.model_dump(mode="json") for row in candidate.prior_searches]
                        ),
                    }
                )
        return path

    def integrity_summary(self) -> dict[str, Any]:
        """Backward-compatible alias for the comprehensive validity verifier."""
        return self.validity_summary()

    def validity_summary(self, *, history_manifest: Path | None = None) -> dict[str, Any]:
        """Recompute the complete Hunter validity and provenance contract."""
        from exo_toolkit.hunter_validity import validate_hunter_database

        resolved_manifest = history_manifest
        if resolved_manifest is None:
            candidate = (
                Path(__file__).resolve().parents[2]
                / "data_selection"
                / "hunter_prior_search_history_v1.json"
            )
            resolved_manifest = candidate if candidate.is_file() else None
        return validate_hunter_database(
            self.db_path,
            schema_version=HUNTER_SCHEMA_VERSION,
            history_manifest=resolved_manifest,
        )
