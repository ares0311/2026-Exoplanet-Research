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

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["new", "follow-up"]
TargetStatus = Literal["candidate_found", "no_signal", "no_data", "failed"]
TERMINAL_TARGET_STATUSES = frozenset({"candidate_found", "no_signal", "no_data"})
EXECUTABLE_SEARCH_STATES = frozenset({"pending", "running", "partial", "failed"})
HUNTER_SCHEMA_VERSION = 1


class PriorSearch(BaseModel):
    """One provenance-complete search performed by any reliable project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    searched_by: str = Field(min_length=1)
    searched_at: datetime
    source_project: str = Field(min_length=1)
    method_or_data: str = Field(min_length=1)
    result: str = Field(min_length=1)
    provenance_uri: str = Field(min_length=1)


class HunterCandidate(BaseModel):
    """Normalized candidate-universe row frozen into a search snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str = Field(min_length=1)
    canonical_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    mission: Literal["TESS", "Kepler", "K2", "JWST"] = "TESS"
    object_classification: str = Field(default="star", min_length=1)
    source: str = Field(min_length=1)
    source_provenance: dict[str, Any]
    eligible: bool = True
    eligibility_reason: str = Field(default="eligible", min_length=1)
    distance_pc: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    estimated_download_gb: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    ranking_score: float = Field(allow_inf_nan=False)
    selection_reason: str = Field(min_length=1)
    metrics: dict[str, float | int | str | None]
    prior_searches: tuple[PriorSearch, ...] = ()

    @model_validator(mode="after")
    def follow_up_rows_require_history(self) -> HunterCandidate:
        if self.source_provenance.get("search_category") == "follow-up" and not self.prior_searches:
            raise ValueError("follow-up candidates require prior_searches provenance")
        return self


class FollowUpRecommendation(BaseModel):
    """Validated evidence and action for one follow-up registry entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    priority: float = Field(allow_inf_nan=False)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any]
    recommended_action: str = Field(min_length=1)


class TargetExecutionResult(BaseModel):
    """One target's complete acquisition-through-interpretation outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: TargetStatus
    result: dict[str, Any]
    provenance: dict[str, Any]
    error_message: str | None = None
    follow_ups: tuple[FollowUpRecommendation, ...] = ()

    @model_validator(mode="after")
    def failure_requires_error(self) -> TargetExecutionResult:
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed target result requires error_message")
        return self


class SearchExecutionSummary(BaseModel):
    """Structured result returned after one resumable run attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    search_id: str
    attempt_id: str
    status: Literal["completed", "partial", "failed"]
    started_at: datetime
    completed_at: datetime
    targets_total: int
    targets_already_complete: int
    targets_processed: int
    targets_succeeded: int
    targets_failed: int
    follow_ups_registered: int


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
    mode TEXT NOT NULL,
    requested_target_count INTEGER NOT NULL,
    selected_target_count INTEGER NOT NULL,
    candidate_pool_count INTEGER NOT NULL,
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
    state TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS search_runs (
    attempt_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    config_json TEXT NOT NULL,
    items_processed INTEGER NOT NULL DEFAULT 0,
    items_succeeded INTEGER NOT NULL DEFAULT 0,
    items_failed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS target_search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL,
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
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(search_id) REFERENCES search_manifests(search_id)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);
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
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations "
                "(version, applied_at, description) VALUES (?, ?, ?)",
                (
                    HUNTER_SCHEMA_VERSION,
                    _utc_now().isoformat(),
                    "Hunter durable search lifecycle",
                ),
            )
            connection.execute(f"PRAGMA user_version = {HUNTER_SCHEMA_VERSION}")

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
        if len(eligible) < requested_target_count:
            raise RuntimeError(
                f"Requested {requested_target_count} {mode} targets but only "
                f"{len(eligible)} eligible candidates are available; no search was created"
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
                return (
                    raw
                    if isinstance(raw, TargetExecutionResult)
                    else TargetExecutionResult.model_validate(raw)
                )
            except Exception as exc:  # noqa: BLE001
                return TargetExecutionResult(
                    status="failed",
                    result={},
                    provenance={"runner": getattr(runner, "__name__", type(runner).__name__)},
                    error_message=f"{type(exc).__name__}: {exc}",
                )

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_safe_run, candidate): candidate for candidate in pending}
                for done, future in enumerate(as_completed(futures), 1):
                    candidate = futures[future]
                    outcome = future.result()
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
                                _canonical_json(outcome.provenance),
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

    @staticmethod
    def _register_follow_up(
        connection: sqlite3.Connection,
        *,
        search_id: str,
        candidate: HunterCandidate,
        follow_up: FollowUpRecommendation,
        created_at: datetime,
    ) -> bool:
        follow_up_id = hashlib.sha256(
            f"{search_id}:{candidate.target_id}:{follow_up.candidate_id}".encode()
        ).hexdigest()
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO follow_up_registry (
                follow_up_id, search_id, target_id, candidate_id, priority, reason,
                evidence_json, prior_search_provenance_json, recommended_action,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                follow_up_id,
                search_id,
                candidate.target_id,
                follow_up.candidate_id,
                follow_up.priority,
                follow_up.reason,
                _canonical_json(follow_up.evidence),
                _canonical_json([row.model_dump(mode="json") for row in candidate.prior_searches]),
                follow_up.recommended_action,
                created_at.isoformat(),
            ),
        )
        return cursor.rowcount == 1

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
        return result

    def follow_up_candidates(self) -> list[HunterCandidate]:
        candidates: list[HunterCandidate] = []
        for row in self.list_follow_ups():
            prior = PriorSearch(
                searched_by="EXO-Hunter",
                searched_at=datetime.fromisoformat(str(row["created_at"])),
                source_project="2026 Exoplanet Research",
                method_or_data="EXO-Hunter acquisition, BLS vetting, and scoring pipeline",
                result=str(row["reason"]),
                provenance_uri=f"hunter-search:{row['search_id']}",
            )
            candidates.append(
                HunterCandidate(
                    target_id=str(row["target_id"]),
                    canonical_id=str(row["target_id"]),
                    source="EXO-Hunter follow-up registry",
                    source_provenance={
                        "search_category": "follow-up",
                        "follow_up_id": row["follow_up_id"],
                        "search_id": row["search_id"],
                    },
                    ranking_score=float(row["priority"]),
                    selection_reason=str(row["reason"]),
                    metrics={"follow_up_priority": float(row["priority"])},
                    prior_searches=(prior,),
                )
            )
        return candidates

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
        required_tables = {
            "candidate_catalog",
            "search_manifests",
            "search_manifest_targets",
            "search_state_events",
            "search_runs",
            "target_search_history",
            "follow_up_registry",
        }
        with self.connect() as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            orphan_targets = int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_manifest_targets AS t "
                    "LEFT JOIN candidate_catalog AS c ON c.snapshot_id=t.snapshot_id "
                    "WHERE c.snapshot_id IS NULL"
                ).fetchone()[0]
            )
        missing = sorted(required_tables - tables)
        return {
            "ok": integrity == "ok" and not missing and orphan_targets == 0,
            "sqlite_integrity": integrity,
            "missing_tables": missing,
            "orphan_manifest_targets": orphan_targets,
            "schema_version": HUNTER_SCHEMA_VERSION,
        }
