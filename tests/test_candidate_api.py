"""Tests for Skills.candidate_api."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from Skills.candidate_api import (
    CandidateAPI,
    api_response,
    background_summary_payload,
    candidate_to_payload,
    summary_payload,
)


def _row(
    *,
    candidate_id: str = "TIC1-s01",
    target_id: str = "TIC 1",
    fpp: float | None = 0.10,
    pathway: str = "planet_hunters_discussion",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "period_days": 3.5,
        "depth_ppm": 900.0,
        "snr": 12.0,
        "scores": {
            "false_positive_probability": fpp,
            "detection_confidence": 0.82,
        },
        "recommended_pathway": pathway,
        "explanation": {
            "positive_evidence": ["Repeated candidate signal"],
            "negative_evidence": ["False-positive evidence not fully excluded"],
            "blocking_issues": [],
        },
    }


def _json_body(body: bytes) -> object:
    return json.loads(body.decode("utf-8"))


def _background_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE run_ledger (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                command TEXT NOT NULL,
                target_id TEXT,
                outcome TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                config_version TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                config_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            );
            CREATE TABLE reviewed_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                target_id TEXT NOT NULL,
                reason_codes_json TEXT NOT NULL,
                negative_evidence_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE needs_follow_up_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                target_id TEXT NOT NULL,
                trigger_codes_json TEXT NOT NULL,
                mandatory_tests_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE report_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                format TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE approval_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                approved INTEGER NOT NULL,
                approver TEXT NOT NULL,
                approval_scope TEXT NOT NULL,
                rationale TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            PRAGMA user_version = 2;
            """
        )
    return db_path


def _insert_run(
    db_path: Path,
    *,
    run_id: str,
    outcome: str,
    status: str = "completed",
    target_id: str = "TIC 1",
    completed_at: str = "2026-05-20T00:00:00+00:00",
    error_message: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO run_ledger (
                run_id, started_at, completed_at, command, target_id, outcome, status,
                error_message, config_version, config_fingerprint, config_json, provenance_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                "2026-05-20T00:00:00+00:00",
                completed_at,
                "exo background-run-once",
                target_id,
                outcome,
                status,
                error_message,
                "background_search_v0",
                "a" * 64,
                "{}",
                "[]",
            ),
        )


def test_candidate_api_sorts_by_fpp() -> None:
    api = CandidateAPI([_row(candidate_id="high", fpp=0.8), _row(candidate_id="low", fpp=0.02)])
    assert [candidate.candidate_id for candidate in api.candidates] == ["low", "high"]


def test_candidate_by_id_returns_candidate() -> None:
    api = CandidateAPI([_row(candidate_id="TIC1-s01")])
    candidate = api.candidate_by_id("TIC1-s01")
    assert candidate is not None
    assert candidate.target_id == "TIC 1"


def test_candidate_by_id_missing_returns_none() -> None:
    api = CandidateAPI([_row(candidate_id="TIC1-s01")])
    assert api.candidate_by_id("missing") is None


def test_candidate_payload_is_json_ready() -> None:
    api = CandidateAPI([_row()])
    payload = candidate_to_payload(api.candidates[0])
    assert payload["positive_evidence"] == ["Repeated candidate signal"]
    assert payload["language_guardrail"].startswith("candidate signal")


def test_summary_payload_marks_read_only_and_no_live_services() -> None:
    api = CandidateAPI([_row()])
    payload = summary_payload(api.candidates)
    assert payload["read_only"] is True
    assert payload["live_services"] is False
    assert payload["external_submission"] is False


def test_summary_payload_counts_blocked_rows() -> None:
    row = _row(fpp=0.01)
    row["explanation"]["blocking_issues"] = ["Missing provenance"]
    api = CandidateAPI([row])
    payload = summary_payload(api.candidates)
    assert payload["blocked_count"] == 1
    assert payload["risk_counts"]["blocked"] == 1


def test_health_endpoint() -> None:
    status, content_type, body = api_response(CandidateAPI([_row()]), "/health")
    payload = _json_body(body)
    assert status == 200
    assert content_type.startswith("application/json")
    assert payload["ok"] is True


def test_summary_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row()]), "/summary")
    payload = _json_body(body)
    assert status == 200
    assert payload["n_candidates"] == 1


def test_candidates_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates")
    payload = _json_body(body)
    assert status == 200
    assert payload[0]["candidate_id"] == "A"


def test_candidate_detail_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates/A")
    payload = _json_body(body)
    assert status == 200
    assert payload["candidate_id"] == "A"


def test_candidate_detail_endpoint_decodes_url() -> None:
    api = CandidateAPI([_row(candidate_id="TIC 1/s01")])
    status, _, body = api_response(api, "/candidates/TIC%201%2Fs01")
    payload = _json_body(body)
    assert status == 200
    assert payload["candidate_id"] == "TIC 1/s01"


def test_missing_candidate_endpoint_returns_404() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates/B")
    payload = _json_body(body)
    assert status == 404
    assert payload["error"] == "candidate not found"


def test_dashboard_endpoint_returns_html() -> None:
    status, content_type, body = api_response(CandidateAPI([_row()]), "/dashboard")
    text = body.decode("utf-8")
    assert status == 200
    assert content_type.startswith("text/html")
    assert "does not claim a confirmed planet" in text


def test_root_endpoint_lists_endpoints() -> None:
    status, _, body = api_response(CandidateAPI([]), "/")
    payload = _json_body(body)
    assert status == 200
    assert "/candidates" in payload["endpoints"]


def test_unknown_route_returns_404() -> None:
    status, _, body = api_response(CandidateAPI([]), "/missing")
    payload = _json_body(body)
    assert status == 404
    assert payload["error"] == "not found"


def test_missing_scores_stay_none_not_zero() -> None:
    api = CandidateAPI([{"candidate_id": "missing", "target_id": "TIC 9"}])
    status, _, body = api_response(api, "/candidates/missing")
    payload = _json_body(body)
    assert status == 200
    assert payload["false_positive_probability"] is None


def test_no_confirmed_planet_claim_in_json_payloads() -> None:
    _, _, body = api_response(CandidateAPI([_row()]), "/summary")
    assert "confirmed planet" not in body.decode("utf-8").lower()


def test_known_object_row_is_still_review_language() -> None:
    row = _row(candidate_id="known")
    row["recommended_pathway"] = "known_object_annotation"
    row["status"] = "candidate_signal"
    api = CandidateAPI([row])
    payload = candidate_to_payload(api.candidates[0])
    assert payload["pathway"] == "known_object_annotation"
    assert "candidate signal" in payload["language_guardrail"]


def test_background_summary_without_configured_db_is_unavailable() -> None:
    payload = background_summary_payload(None)
    assert payload["available"] is False
    assert payload["read_only"] is True
    assert payload["reason"] == "background SQLite path is not configured"


def test_background_summary_missing_db_does_not_create_file(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    payload = background_summary_payload(db_path)
    assert payload["available"] is False
    assert payload["reason"] == "background SQLite database does not exist"
    assert not db_path.exists()


def test_background_summary_empty_db(tmp_path: Path) -> None:
    db_path = _background_db(tmp_path)
    payload = background_summary_payload(db_path)
    assert payload["available"] is True
    assert payload["ledger_count"] == 0
    assert payload["latest_run"] is None
    assert payload["latest_alert"] is False


def test_background_summary_reviewed_latest_run(tmp_path: Path) -> None:
    db_path = _background_db(tmp_path)
    _insert_run(db_path, run_id="run_reviewed", outcome="reviewed")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO reviewed_outcomes (
                run_id, target_id, reason_codes_json, negative_evidence_json, summary, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "run_reviewed",
                "TIC 1",
                '["no_follow_up_triggers"]',
                '["weak evidence"]',
                "Reviewed with no follow-up triggers.",
                "2026-05-20T00:00:00+00:00",
            ),
        )

    payload = background_summary_payload(db_path)
    assert payload["reviewed_count"] == 1
    assert payload["latest_run"]["outcome"] == "reviewed"
    assert payload["latest_alert"] is False
    assert payload["latest_reason"] == "Reviewed with no follow-up triggers."


def test_background_summary_needs_follow_up_exposes_reports_and_approval(
    tmp_path: Path,
) -> None:
    db_path = _background_db(tmp_path)
    _insert_run(db_path, run_id="run_followup", outcome="needs_follow_up")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO needs_follow_up_outcomes (
                run_id, target_id, trigger_codes_json, mandatory_tests_json, summary, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "run_followup",
                "TIC 1",
                '["high_priority_score"]',
                "[]",
                "Fixture target requires follow-up record and human-readable review.",
                "2026-05-20T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO report_exports (run_id, target_id, format, path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run_followup",
                "TIC 1",
                "markdown",
                "reports/background/run_followup.md",
                "2026-05-20T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO approval_records (
                run_id, target_id, approved, approver, approval_scope, rationale, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_followup",
                "TIC 1",
                0,
                "system",
                "external_submission",
                "External submission remains blocked.",
                "2026-05-20T00:00:00+00:00",
            ),
        )

    payload = background_summary_payload(db_path)
    assert payload["needs_follow_up_count"] == 1
    assert payload["latest_alert"] is True
    assert payload["latest_report_paths"] == ["reports/background/run_followup.md"]
    assert payload["latest_approval"]["external_submission_approved"] is False
    assert payload["external_submission"] is False


def test_background_summary_blocked_latest_run_alerts(tmp_path: Path) -> None:
    db_path = _background_db(tmp_path)
    _insert_run(
        db_path,
        run_id="run_blocked",
        outcome="blocked",
        status="blocked",
        target_id="RUN_LOCK",
        error_message="Another background run still owns the run lock.",
    )

    payload = background_summary_payload(db_path)
    assert payload["latest_run"]["status"] == "blocked"
    assert payload["latest_alert"] is True
    assert payload["latest_reason"] == "Another background run still owns the run lock."
    assert payload["outcome_counts"]["blocked"] == 1


def test_background_latest_endpoint_returns_read_only_payload(tmp_path: Path) -> None:
    db_path = _background_db(tmp_path)
    _insert_run(db_path, run_id="run_latest", outcome="reviewed")
    api = CandidateAPI([], background_db_path=db_path)

    status, _, body = api_response(api, "/background/latest")
    payload = _json_body(body)

    assert status == 200
    assert payload["available"] is True
    assert payload["latest_run"]["run_id"] == "run_latest"
    assert payload["read_only"] is True


def test_root_endpoint_lists_background_endpoints() -> None:
    status, _, body = api_response(CandidateAPI([]), "/")
    payload = _json_body(body)
    assert status == 200
    assert "/background/summary" in payload["endpoints"]
    assert "/background/latest" in payload["endpoints"]
