"""SQLite persistence for background search automation."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from exo_toolkit.background.reason_codes import STABLE_REASON_CODES

DB_SCHEMA_VERSION = 2


class ClosingConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


class BackgroundStore:
    """Durable SQLite store rooted at the top-level logs directory by default."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0, factory=ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_ledger (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    command TEXT NOT NULL,
                    target_id TEXT,
                    outcome TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    config_version TEXT NOT NULL DEFAULT 'unknown',
                    config_fingerprint TEXT NOT NULL DEFAULT 'unknown',
                    config_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS target_priority_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    factors_json TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    selected INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS reviewed_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    negative_evidence_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS needs_follow_up_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL,
                    trigger_codes_json TEXT NOT NULL,
                    mandatory_tests_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS follow_up_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS draft_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL,
                    ready INTEGER NOT NULL,
                    sections_json TEXT NOT NULL,
                    blocking_issues_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS submission_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    suitability_rationale TEXT NOT NULL,
                    risks_json TEXT NOT NULL,
                    prerequisites_json TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    human_approval_required INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS report_exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    format TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS approval_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    approver TEXT NOT NULL,
                    approval_scope TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES run_ledger(run_id)
                );

                CREATE TABLE IF NOT EXISTS run_locks (
                    lock_name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    acquired_at TEXT NOT NULL
                );
                """
            )
            _ensure_column(
                connection,
                "run_ledger",
                "config_version",
                "TEXT NOT NULL DEFAULT 'unknown'",
            )
            _ensure_column(
                connection,
                "run_ledger",
                "config_fingerprint",
                "TEXT NOT NULL DEFAULT 'unknown'",
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, applied_at, description)
                VALUES (?, datetime('now'), ?)
                """,
                (DB_SCHEMA_VERSION, "background automation SQLite schema"),
            )
            connection.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION}")

    def acquire_run_lock(
        self,
        lock_name: str,
        owner: str,
        wait_seconds: float,
        poll_seconds: float,
        acquired_at: str,
    ) -> bool:
        deadline = time.monotonic() + wait_seconds
        while True:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO run_locks (lock_name, owner, acquired_at)
                    VALUES (?, ?, ?)
                    """,
                    (lock_name, owner, acquired_at),
                )
                if cursor.rowcount == 1:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_seconds)

    def release_run_lock(self, lock_name: str, owner: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM run_locks WHERE lock_name = ? AND owner = ?", (lock_name, owner)
            )

    def has_seen_target(self, target_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM run_ledger WHERE target_id = ? LIMIT 1", (target_id,)
            ).fetchone()
        return row is not None

    def write_run_ledger(self, entry: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_ledger (
                    run_id, started_at, completed_at, command, target_id, outcome, status,
                    error_message, config_version, config_fingerprint, config_json, provenance_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["run_id"],
                    entry["started_at"],
                    entry["completed_at"],
                    entry["command"],
                    entry.get("target_id"),
                    entry["outcome"],
                    entry["status"],
                    entry.get("error_message"),
                    entry["config"]["config_version"],
                    entry["config"]["config_fingerprint"],
                    _to_json(entry["config"]),
                    _to_json(entry["provenance"]),
                ),
            )

    def write_priority_evaluations(
        self, run_id: str, evaluations: list[dict[str, Any]], created_at: str
    ) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO target_priority_evaluations (
                    run_id, target_id, target_name, factors_json, final_score,
                    reason_codes_json, selected, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        evaluation["target_id"],
                        evaluation["target_name"],
                        _to_json(evaluation["factors"]),
                        evaluation["final_score"],
                        _to_json(evaluation["reason_codes"]),
                        int(evaluation["selected"]),
                        created_at,
                    )
                    for evaluation in evaluations
                ],
            )

    def write_reviewed_outcome(self, entry: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO reviewed_outcomes (
                    run_id, target_id, reason_codes_json, negative_evidence_json,
                    summary, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["run_id"],
                    entry["target_id"],
                    _to_json(entry["reason_codes"]),
                    _to_json(entry["negative_evidence"]),
                    entry["summary"],
                    entry["created_at"],
                ),
            )

    def write_needs_follow_up_outcome(self, entry: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO needs_follow_up_outcomes (
                    run_id, target_id, trigger_codes_json, mandatory_tests_json,
                    summary, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["run_id"],
                    entry["target_id"],
                    _to_json(entry["trigger_codes"]),
                    _to_json(entry["mandatory_tests"]),
                    entry["summary"],
                    entry["created_at"],
                ),
            )

    def write_follow_up_tests(
        self, run_id: str, target_id: str, tests: list[dict[str, str]], created_at: str
    ) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO follow_up_tests (
                    run_id, target_id, test_name, status, rationale, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        target_id,
                        test["test_name"],
                        test["status"],
                        test["rationale"],
                        created_at,
                    )
                    for test in tests
                ],
            )

    def write_draft_report(self, run_id: str, report: dict[str, Any], created_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO draft_reports (
                    run_id, target_id, ready, sections_json, blocking_issues_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    report["target_id"],
                    int(report["ready"]),
                    _to_json(report["sections"]),
                    _to_json(report["blocking_issues"]),
                    created_at,
                ),
            )

    def write_submission_recommendations(
        self, run_id: str, target_id: str, recommendations: list[dict[str, Any]], created_at: str
    ) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO submission_recommendations (
                    run_id, target_id, destination, rank, suitability_rationale, risks_json,
                    prerequisites_json, recommended_action, human_approval_required, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        target_id,
                        recommendation["destination"],
                        recommendation["rank"],
                        recommendation["suitability_rationale"],
                        _to_json(recommendation["risks"]),
                        _to_json(recommendation["prerequisites"]),
                        recommendation["recommended_action"],
                        int(recommendation["human_approval_required"]),
                        created_at,
                    )
                    for recommendation in recommendations
                ],
            )

    def write_report_exports(
        self, run_id: str, target_id: str, exports: dict[str, str], created_at: str
    ) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO report_exports (run_id, target_id, format, path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (run_id, target_id, export_format, path, created_at)
                    for export_format, path in exports.items()
                ],
            )

    def write_approval_record(self, run_id: str, entry: dict[str, Any], created_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO approval_records (
                    run_id, target_id, approved, approver, approval_scope, rationale, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    entry["target_id"],
                    int(entry["approved"]),
                    entry["approver"],
                    entry["approval_scope"],
                    entry["rationale"],
                    created_at,
                ),
            )

    def list_run_ledger(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM run_ledger ORDER BY started_at ASC").fetchall()
        return [_decode_row(dict(row)) for row in rows]

    def list_reviewed_outcomes(self) -> list[dict[str, Any]]:
        return self._list_table("reviewed_outcomes")

    def list_needs_follow_up_outcomes(self) -> list[dict[str, Any]]:
        return self._list_table("needs_follow_up_outcomes")

    def list_follow_up_tests(self) -> list[dict[str, Any]]:
        return self._list_table("follow_up_tests")

    def list_draft_reports(self) -> list[dict[str, Any]]:
        return self._list_table("draft_reports")

    def list_submission_recommendations(self) -> list[dict[str, Any]]:
        return self._list_table("submission_recommendations")

    def list_report_exports(self) -> list[dict[str, Any]]:
        return self._list_table("report_exports")

    def list_approval_records(self) -> list[dict[str, Any]]:
        return self._list_table("approval_records")

    def run_summary(self, run_id: str | None = None) -> dict[str, Any]:
        run = self._fetch_run(run_id)
        if run is None:
            return {"run": None}
        resolved_run_id = run["run_id"]
        return {
            "run": run,
            "priority_evaluations": self._list_table_for_run(
                "target_priority_evaluations", resolved_run_id
            ),
            "reviewed_outcome": self._list_table_for_run("reviewed_outcomes", resolved_run_id),
            "needs_follow_up_outcome": self._list_table_for_run(
                "needs_follow_up_outcomes", resolved_run_id
            ),
            "follow_up_tests": self._list_table_for_run("follow_up_tests", resolved_run_id),
            "draft_report": self._list_table_for_run("draft_reports", resolved_run_id),
            "submission_recommendations": self._list_table_for_run(
                "submission_recommendations", resolved_run_id
            ),
            "report_exports": self._list_table_for_run("report_exports", resolved_run_id),
            "approval_records": self._list_table_for_run("approval_records", resolved_run_id),
        }

    def target_history(self, target_id: str) -> dict[str, Any]:
        return {
            "target_id": target_id,
            "runs": self._list_table_for_target(
                "run_ledger", target_id, order_column="completed_at"
            ),
            "priority_evaluations": self._list_table_for_target(
                "target_priority_evaluations", target_id
            ),
            "reviewed_outcomes": self._list_table_for_target("reviewed_outcomes", target_id),
            "needs_follow_up_outcomes": self._list_table_for_target(
                "needs_follow_up_outcomes", target_id
            ),
            "follow_up_tests": self._list_table_for_target("follow_up_tests", target_id),
            "approval_records": self._list_table_for_target("approval_records", target_id),
        }

    def scheduler_notification_summary(self, run_id: str | None = None) -> dict[str, Any]:
        summary = self.run_summary(run_id)
        run = summary["run"]
        if run is None:
            return {
                "status": "empty",
                "alert": False,
                "message": "No background runs are recorded.",
            }
        outcome = run["outcome"]
        alert = outcome in {"needs_follow_up", "blocked"} or run["status"] != "completed"
        report_paths = [export["path"] for export in summary["report_exports"]]
        return {
            "status": run["status"],
            "outcome": outcome,
            "alert": alert,
            "run_id": run["run_id"],
            "target_id": run["target_id"],
            "reason": run.get("error_message") or _summary_reason(summary),
            "report_paths": report_paths,
        }

    def validation_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            ledger_count = _count(connection, "run_ledger")
            reviewed_count = _count(connection, "reviewed_outcomes")
            needs_follow_up_count = _count(connection, "needs_follow_up_outcomes")
            outcome_count = reviewed_count + needs_follow_up_count
            recommendation_count = _count(connection, "submission_recommendations")
            approval_count = _count(connection, "approval_records")
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        integrity = self.integrity_summary()
        return {
            "database_path": str(self.db_path),
            "sqlite_schema_version": schema_version,
            "ledger_count": ledger_count,
            "reviewed_count": reviewed_count,
            "needs_follow_up_count": needs_follow_up_count,
            "outcome_count": outcome_count,
            "exactly_one_outcome_per_run": ledger_count == outcome_count,
            "submission_recommendation_count": recommendation_count,
            "approval_record_count": approval_count,
            "integrity_ok": integrity["ok"],
            "integrity": integrity,
            "human_approval_gate": "external submission commands are not implemented",
        }

    def integrity_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            runs = {
                row["run_id"]
                for row in connection.execute("SELECT run_id FROM run_ledger").fetchall()
            }
            reviewed = {
                row["run_id"]
                for row in connection.execute("SELECT run_id FROM reviewed_outcomes").fetchall()
            }
            needs = {
                row["run_id"]
                for row in connection.execute(
                    "SELECT run_id FROM needs_follow_up_outcomes"
                ).fetchall()
            }
            missing_config = _count_where(
                connection,
                "run_ledger",
                (
                    "config_version = 'unknown' OR config_fingerprint = 'unknown' "
                    "OR config_fingerprint = ''"
                ),
            )
            orphan_counts = {
                table: _orphan_count(connection, table)
                for table in [
                    "target_priority_evaluations",
                    "reviewed_outcomes",
                    "needs_follow_up_outcomes",
                    "follow_up_tests",
                    "draft_reports",
                    "submission_recommendations",
                    "report_exports",
                    "approval_records",
                ]
            }
            invalid_reason_codes = _invalid_reason_codes(connection)
        outcome_counts = {
            run_id: int(run_id in reviewed) + int(run_id in needs)
            for run_id in runs | reviewed | needs
        }
        bad_outcome_runs = [
            run_id for run_id, outcome_count in outcome_counts.items() if outcome_count != 1
        ]
        ok = (
            not bad_outcome_runs
            and missing_config == 0
            and all(count == 0 for count in orphan_counts.values())
            and not invalid_reason_codes
        )
        return {
            "ok": ok,
            "bad_outcome_runs": sorted(bad_outcome_runs),
            "missing_config_fingerprint_count": missing_config,
            "orphan_counts": orphan_counts,
            "invalid_reason_codes": invalid_reason_codes,
        }

    def _list_table(self, table: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()
        return [_decode_row(dict(row)) for row in rows]

    def _list_table_for_run(self, table: str, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {_order_column(table)} ASC",
                (run_id,),
            ).fetchall()
        return [_decode_row(dict(row)) for row in rows]

    def _list_table_for_target(
        self, table: str, target_id: str, order_column: str | None = None
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            order = order_column or _order_column(table)
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE target_id = ? ORDER BY {order} ASC",
                (target_id,),
            ).fetchall()
        return [_decode_row(dict(row)) for row in rows]

    def _fetch_run(self, run_id: str | None) -> dict[str, Any] | None:
        with self.connect() as connection:
            if run_id is None:
                row = connection.execute(
                    """
                    SELECT * FROM run_ledger
                    ORDER BY completed_at DESC, started_at DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM run_ledger WHERE run_id = ?", (run_id,)
                ).fetchone()
        return None if row is None else _decode_row(dict(row))


def _count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def _count_where(connection: sqlite3.Connection, table: str, where_clause: str) -> int:
    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {where_clause}"
    ).fetchone()
    return int(row["count"])


def _orphan_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM {table}
        LEFT JOIN run_ledger ON {table}.run_id = run_ledger.run_id
        WHERE run_ledger.run_id IS NULL
        """
    ).fetchone()
    return int(row["count"])


def _invalid_reason_codes(connection: sqlite3.Connection) -> dict[str, list[str]]:
    invalid: dict[str, list[str]] = {}
    checks = {
        "target_priority_evaluations.reason_codes": (
            "SELECT reason_codes_json AS codes FROM target_priority_evaluations",
            "codes",
        ),
        "reviewed_outcomes.reason_codes": (
            "SELECT reason_codes_json AS codes FROM reviewed_outcomes",
            "codes",
        ),
        "needs_follow_up_outcomes.trigger_codes": (
            "SELECT trigger_codes_json AS codes FROM needs_follow_up_outcomes",
            "codes",
        ),
    }
    for label, (query, column) in checks.items():
        codes: set[str] = set()
        for row in connection.execute(query).fetchall():
            codes.update(json.loads(row[column]))
        unknown = sorted(codes - STABLE_REASON_CODES)
        if unknown:
            invalid[label] = unknown
    return invalid


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, column_definition: str
) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_definition}")


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if key.endswith("_json") and isinstance(value, str):
            decoded[key.removesuffix("_json")] = json.loads(value)
        else:
            decoded[key] = value
    return decoded


def _order_column(table: str) -> str:
    if table == "run_ledger":
        return "completed_at"
    return "id"


def _summary_reason(summary: dict[str, Any]) -> str:
    needs = summary["needs_follow_up_outcome"]
    if needs:
        return ", ".join(needs[0].get("trigger_codes", []))
    reviewed = summary["reviewed_outcome"]
    if reviewed:
        return ", ".join(reviewed[0].get("reason_codes", []))
    return ""
