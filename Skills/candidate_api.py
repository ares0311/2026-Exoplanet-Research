"""Serve local candidate rows through a read-only standard-library API.

The API loads existing local JSON candidate rows, normalizes them through the
dashboard contract, and exposes JSON plus an optional static dashboard HTML
view. It never queries live services and never mutates source data.

Public API
----------
CandidateAPI(rows, *, title, source_label, background_db_path)
candidate_to_payload(candidate) -> dict[str, Any]
summary_payload(candidates) -> dict[str, Any]
background_summary_payload(db_path) -> dict[str, Any]
artifact_payload(api) -> dict[str, Any]
api_response(api, path) -> tuple[int, str, bytes]
run_server(rows, *, host, port, title, source_label, background_db_path) -> None
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from Skills.candidate_dashboard_export import (
    DashboardCandidate,
    build_dashboard,
    load_dashboard_rows,
    normalize_candidate,
)


class CandidateAPI:
    """Read-only candidate API state."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        title: str = "Candidate Review Dashboard",
        source_label: str = "local JSON",
        background_db_path: Path | None = None,
    ) -> None:
        self.rows = list(rows)
        self.title = title
        self.source_label = source_label
        self.background_db_path = background_db_path
        self.candidates = sorted(
            (normalize_candidate(row) for row in self.rows),
            key=lambda c: (
                c.false_positive_probability is None,
                (
                    c.false_positive_probability
                    if c.false_positive_probability is not None
                    else 1.0
                ),
                c.candidate_id,
            ),
        )

    def candidate_by_id(self, candidate_id: str) -> DashboardCandidate | None:
        """Return a normalized candidate by ID, or None."""
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None


def candidate_to_payload(candidate: DashboardCandidate) -> dict[str, Any]:
    """Convert a dashboard candidate to a JSON-serializable payload."""
    payload = asdict(candidate)
    payload["positive_evidence"] = list(candidate.positive_evidence)
    payload["negative_evidence"] = list(candidate.negative_evidence)
    payload["blocking_issues"] = list(candidate.blocking_issues)
    payload["language_guardrail"] = (
        "candidate signal; no discovery or external validation claim"
    )
    return payload


def summary_payload(candidates: list[DashboardCandidate]) -> dict[str, Any]:
    """Build aggregate summary payload for normalized candidates."""
    fpps = [
        candidate.false_positive_probability
        for candidate in candidates
        if candidate.false_positive_probability is not None
    ]
    risk_counts: dict[str, int] = {}
    pathway_counts: dict[str, int] = {}
    for candidate in candidates:
        risk_counts[candidate.risk_band] = risk_counts.get(candidate.risk_band, 0) + 1
        pathway_counts[candidate.pathway] = pathway_counts.get(candidate.pathway, 0) + 1
    return {
        "n_candidates": len(candidates),
        "min_false_positive_probability": min(fpps) if fpps else None,
        "median_false_positive_probability": (
            sorted(fpps)[len(fpps) // 2] if fpps else None
        ),
        "blocked_count": risk_counts.get("blocked", 0),
        "risk_counts": risk_counts,
        "pathway_counts": pathway_counts,
        "read_only": True,
        "live_services": False,
        "external_submission": False,
        "language_guardrail": (
            "candidate signals and follow-up targets only; no discovery claims"
        ),
    }


def background_summary_payload(db_path: Path | None) -> dict[str, Any]:
    """Build a read-only summary for a background SQLite runtime log."""
    base: dict[str, Any] = {
        "database_path": str(db_path) if db_path is not None else None,
        "available": False,
        "read_only": True,
        "live_services": False,
        "external_submission": False,
        "language_guardrail": (
            "background runs may identify follow-up targets only; no discovery claims"
        ),
    }
    if db_path is None:
        return {**base, "reason": "background SQLite path is not configured"}
    if not db_path.exists():
        return {**base, "reason": "background SQLite database does not exist"}

    try:
        with _connect_read_only(db_path) as connection:
            tables = _sqlite_tables(connection)
            ledger_count = _count_if_table(connection, tables, "run_ledger")
            reviewed_count = _count_if_table(connection, tables, "reviewed_outcomes")
            needs_follow_up_count = _count_if_table(
                connection, tables, "needs_follow_up_outcomes"
            )
            latest_run = _latest_run(connection, tables)
            latest_run_id = latest_run["run_id"] if latest_run is not None else None
            latest_report_paths = _report_paths(connection, tables, latest_run_id)
            latest_approval = _approval_summary(connection, tables, latest_run_id)
            reason = _latest_reason(connection, tables, latest_run)
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            outcome_counts = _outcome_counts(connection, tables)
    except sqlite3.Error as exc:
        return {
            **base,
            "reason": f"could not read background SQLite database: {exc}",
        }

    alert = False
    if latest_run is not None:
        alert = latest_run["outcome"] in {"needs_follow_up", "blocked"} or (
            latest_run["status"] != "completed"
        )
    return {
        **base,
        "available": True,
        "reason": None,
        "sqlite_schema_version": schema_version,
        "ledger_count": ledger_count,
        "reviewed_count": reviewed_count,
        "needs_follow_up_count": needs_follow_up_count,
        "outcome_count": reviewed_count + needs_follow_up_count,
        "outcome_counts": outcome_counts,
        "latest_run": latest_run,
        "latest_alert": alert,
        "latest_reason": reason,
        "latest_report_paths": latest_report_paths,
        "latest_approval": latest_approval,
    }


def artifact_payload(api: CandidateAPI) -> dict[str, Any]:
    """Build a single-file JSON artifact for local review bundles."""
    return {
        "artifact_type": "candidate_api_bundle",
        "schema_version": 1,
        "title": api.title,
        "source_label": api.source_label,
        "read_only": True,
        "live_services": False,
        "external_submission": False,
        "language_guardrail": (
            "candidate signals and follow-up targets only; no discovery claims"
        ),
        "summary": summary_payload(api.candidates),
        "candidates": [
            candidate_to_payload(candidate)
            for candidate in api.candidates
        ],
        "background": background_summary_payload(api.background_db_path),
    }


def _json_response(
    status: int,
    payload: dict[str, Any] | list[dict[str, Any]],
) -> tuple[int, str, bytes]:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def api_response(api: CandidateAPI, path: str) -> tuple[int, str, bytes]:
    """Return ``(status, content_type, body)`` for a local API path."""
    parsed = urlparse(path)
    route = parsed.path.rstrip("/") or "/"

    if route == "/health":
        return _json_response(
            200,
            {
                "ok": True,
                "read_only": True,
                "live_services": False,
                "n_candidates": len(api.candidates),
            },
        )
    if route == "/summary":
        return _json_response(200, summary_payload(api.candidates))
    if route == "/background/summary":
        return _json_response(200, background_summary_payload(api.background_db_path))
    if route == "/background/latest":
        payload = background_summary_payload(api.background_db_path)
        return _json_response(
            200,
            {
                "database_path": payload["database_path"],
                "available": payload["available"],
                "read_only": payload["read_only"],
                "live_services": payload["live_services"],
                "external_submission": payload["external_submission"],
                "latest_run": payload.get("latest_run"),
                "latest_alert": payload.get("latest_alert", False),
                "latest_reason": payload.get("latest_reason"),
                "latest_report_paths": payload.get("latest_report_paths", []),
                "latest_approval": payload.get("latest_approval"),
                "language_guardrail": payload["language_guardrail"],
            },
        )
    if route == "/artifact.json":
        return _json_response(200, artifact_payload(api))
    if route == "/candidates":
        return _json_response(
            200,
            [candidate_to_payload(candidate) for candidate in api.candidates],
        )
    if route.startswith("/candidates/"):
        candidate_id = unquote(route.removeprefix("/candidates/"))
        candidate = api.candidate_by_id(candidate_id)
        if candidate is None:
            return _json_response(
                404,
                {"error": "candidate not found", "candidate_id": candidate_id},
            )
        return _json_response(200, candidate_to_payload(candidate))
    if route == "/dashboard":
        body = build_dashboard(
            api.rows,
            title=api.title,
            source_label=api.source_label,
        ).encode("utf-8")
        return 200, "text/html; charset=utf-8", body
    if route == "/" or route == "/index.html":
        return _json_response(
            200,
            {
                "service": "candidate_api",
                "read_only": True,
                "endpoints": [
                    "/health",
                    "/summary",
                    "/candidates",
                    "/candidates/<candidate_id>",
                    "/dashboard",
                    "/artifact.json",
                    "/background/summary",
                    "/background/latest",
                ],
            },
        )
    return _json_response(404, {"error": "not found", "path": route})


def make_handler(api: CandidateAPI) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to a ``CandidateAPI`` instance."""

    class CandidateAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = api_response(api, self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            body = json.dumps(
                {"error": "method not allowed", "read_only": True},
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(405)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return CandidateAPIHandler


def run_server(
    rows: list[dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "Candidate Review Dashboard",
    source_label: str = "local JSON",
    background_db_path: Path | None = None,
) -> None:
    """Run the local read-only HTTP server until interrupted."""
    api = CandidateAPI(
        rows,
        title=title,
        source_label=source_label,
        background_db_path=background_db_path,
    )
    server = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"Serving read-only candidate API at http://{host}:{port}")
    server.serve_forever()


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_api",
        description="Serve local candidate JSON through a read-only API.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--title", default="Candidate Review Dashboard")
    parser.add_argument("--source-label", default="local JSON")
    parser.add_argument(
        "--background-db-path",
        type=Path,
        default=None,
        help="Optional read-only background SQLite log path.",
    )
    args = parser.parse_args(argv)

    rows = load_dashboard_rows(args.files)
    run_server(
        rows,
        host=args.host,
        port=args.port,
        title=args.title,
        source_label=args.source_label,
        background_db_path=args.background_db_path,
    )
    return 0


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}


def _count_if_table(
    connection: sqlite3.Connection,
    tables: set[str],
    table: str,
) -> int:
    if table not in tables:
        return 0
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _latest_run(
    connection: sqlite3.Connection,
    tables: set[str],
) -> dict[str, Any] | None:
    if "run_ledger" not in tables:
        return None
    row = connection.execute(
        """
        SELECT run_id, started_at, completed_at, command, target_id, outcome, status,
               error_message, config_version, config_fingerprint
        FROM run_ledger
        ORDER BY completed_at DESC, started_at DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row is not None else None


def _outcome_counts(
    connection: sqlite3.Connection,
    tables: set[str],
) -> dict[str, int]:
    if "run_ledger" not in tables:
        return {}
    rows = connection.execute(
        "SELECT outcome, COUNT(*) AS count FROM run_ledger GROUP BY outcome"
    ).fetchall()
    return {row["outcome"]: int(row["count"]) for row in rows}


def _report_paths(
    connection: sqlite3.Connection,
    tables: set[str],
    run_id: str | None,
) -> list[str]:
    if run_id is None or "report_exports" not in tables:
        return []
    rows = connection.execute(
        "SELECT path FROM report_exports WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    return [str(row["path"]) for row in rows]


def _approval_summary(
    connection: sqlite3.Connection,
    tables: set[str],
    run_id: str | None,
) -> dict[str, Any]:
    if run_id is None or "approval_records" not in tables:
        return {
            "records": 0,
            "external_submission_approved": False,
            "approval_required": True,
        }
    rows = connection.execute(
        """
        SELECT approved, approval_scope FROM approval_records
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    external_approved = any(
        bool(row["approved"]) and row["approval_scope"] == "external_submission"
        for row in rows
    )
    return {
        "records": len(rows),
        "external_submission_approved": external_approved,
        "approval_required": not external_approved,
    }


def _latest_reason(
    connection: sqlite3.Connection,
    tables: set[str],
    latest_run: dict[str, Any] | None,
) -> str | None:
    if latest_run is None:
        return None
    if latest_run.get("error_message"):
        return str(latest_run["error_message"])
    run_id = latest_run["run_id"]
    if "needs_follow_up_outcomes" in tables:
        row = connection.execute(
            "SELECT summary FROM needs_follow_up_outcomes WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            return str(row["summary"])
    if "reviewed_outcomes" in tables:
        row = connection.execute(
            "SELECT summary FROM reviewed_outcomes WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            return str(row["summary"])
    return None


if __name__ == "__main__":
    raise SystemExit(_cli())
