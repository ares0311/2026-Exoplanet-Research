from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from exo_toolkit.background.config import DEFAULT_CONFIG_PATH, ConfigError, load_background_config
from exo_toolkit.background.fixtures import load_known_tess_examples
from exo_toolkit.background.priority import build_priority_summary
from exo_toolkit.background.reason_codes import STABLE_REASON_CODES, ReasonCode
from exo_toolkit.background.runner import background_run_once
from exo_toolkit.background.storage import BackgroundStore
from exo_toolkit.cli import EXIT_NEEDS_FOLLOW_UP, main


def test_priority_summary_exposes_component_factors(tmp_path: Path) -> None:
    store = BackgroundStore(tmp_path / "logs" / "background_search.sqlite3")
    config = load_background_config(DEFAULT_CONFIG_PATH)
    summary = build_priority_summary(load_known_tess_examples(), store, config)

    assert summary.selected_target_id is not None
    first = summary.evaluations[0].to_jsonable()
    assert "scientific_interest_score" in first["factors"]
    assert "never_reviewed_boost" in first["factors"]
    assert ReasonCode.KNOWN_TESS_EXAMPLE.value in first["reason_codes"]
    assert "skipped_reason_codes" in first


def test_background_run_once_writes_top_level_sqlite_logs(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    config_path = _temp_config(tmp_path)

    result = background_run_once(db_path=db_path, config_path=config_path)

    assert db_path.exists()
    assert db_path.parent.name == "logs"
    assert result.ledger_written is True
    assert result.outcome_written is True

    with sqlite3.connect(db_path) as connection:
        ledger_count = connection.execute("SELECT COUNT(*) FROM run_ledger").fetchone()[0]
        reviewed_count = connection.execute("SELECT COUNT(*) FROM reviewed_outcomes").fetchone()[0]
        needs_count = connection.execute(
            "SELECT COUNT(*) FROM needs_follow_up_outcomes"
        ).fetchone()[0]
        tests_count = connection.execute("SELECT COUNT(*) FROM follow_up_tests").fetchone()[0]
        recommendations_count = connection.execute(
            "SELECT COUNT(*) FROM submission_recommendations"
        ).fetchone()[0]
        export_count = connection.execute("SELECT COUNT(*) FROM report_exports").fetchone()[0]
        approval_count = connection.execute("SELECT COUNT(*) FROM approval_records").fetchone()[0]
        ledger = connection.execute(
            "SELECT config_version, config_fingerprint FROM run_ledger"
        ).fetchone()

    assert ledger_count == 1
    assert reviewed_count + needs_count == 1
    assert tests_count >= 1
    assert recommendations_count == 3
    assert export_count == 2
    assert approval_count == 1
    assert ledger[0] == "background_search_v0"
    assert len(ledger[1]) == 64


def test_validation_summary_enforces_exactly_one_outcome(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    background_run_once(db_path=db_path, config_path=_temp_config(tmp_path))

    store = BackgroundStore(db_path)
    summary = store.validation_summary()

    assert summary["ledger_count"] == 1
    assert summary["exactly_one_outcome_per_run"] is True
    assert summary["sqlite_schema_version"] >= 2
    assert summary["human_approval_gate"] == "external submission commands are not implemented"


def test_cli_background_run_once_uses_sqlite_path(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"

    exit_code = main(["background-run-once", "--db-path", str(db_path), "--no-report-export"])

    assert exit_code == 0
    assert db_path.exists()


def test_config_validation_rejects_live_network_default(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["network_access"] = True
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_background_config(config_path)


def test_dry_run_does_not_write_sqlite_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"

    result = background_run_once(db_path=db_path, dry_run=True, config_path=_temp_config(tmp_path))

    assert result.ledger_written is False
    assert ReasonCode.DRY_RUN_NO_WRITE.value in result.reason_codes
    store = BackgroundStore(db_path)
    assert store.validation_summary()["ledger_count"] == 0


def test_target_id_can_force_reviewed_edge_case(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"

    result = background_run_once(
        db_path=db_path,
        target_id="SYNTH-TESS-WEAK-001",
        export_reports=False,
        config_path=_temp_config(tmp_path),
    )

    assert result.target_id == "SYNTH-TESS-WEAK-001"
    assert result.outcome.value == "reviewed"
    store = BackgroundStore(db_path)
    assert store.validation_summary()["reviewed_count"] == 1


def test_edge_cases_cover_expected_paths(tmp_path: Path) -> None:
    config_path = _temp_config(tmp_path)
    cases = {
        "SYNTH-TESS-WEAK-001": "reviewed",
        "SYNTH-TESS-CONTAM-001": "needs_follow_up",
        "SYNTH-TESS-NOPROV-001": "needs_follow_up",
        "SYNTH-TESS-CAL-001": "needs_follow_up",
    }
    for target_id, expected_outcome in cases.items():
        db_path = tmp_path / f"{target_id}.sqlite3"
        result = background_run_once(
            db_path=db_path,
            target_id=target_id,
            export_reports=False,
            config_path=config_path,
        )
        assert result.outcome.value == expected_outcome
    no_provenance = background_run_once(
        db_path=tmp_path / "noprovenance.sqlite3",
        target_id="SYNTH-TESS-NOPROV-001",
        export_reports=False,
        config_path=config_path,
    )
    assert ReasonCode.MISSING_PROVENANCE.value in no_provenance.reason_codes


def test_reason_codes_are_stable() -> None:
    assert ReasonCode.NO_FOLLOW_UP_TRIGGERS.value in STABLE_REASON_CODES
    assert ReasonCode.RUN_LOCK_UNAVAILABLE.value in STABLE_REASON_CODES
    assert ReasonCode.TARGET_ID_NOT_FOUND.value in STABLE_REASON_CODES


def test_report_guardrail_language(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    result = background_run_once(db_path=db_path, config_path=_temp_config(tmp_path))

    assert result.draft_report is not None
    text = " ".join(result.draft_report.sections.values()).lower()
    assert "confirmed planet" not in text
    assert "new discovery" in text
    assert "does not claim" in text

    assert result.draft_report.export_paths
    markdown = (
        Path(result.draft_report.export_paths["markdown"]).read_text(encoding="utf-8").lower()
    )
    html = Path(result.draft_report.export_paths["html"]).read_text(encoding="utf-8").lower()
    assert "confirmed planet" not in markdown
    assert "confirmed planet" not in html
    assert "does not claim" in markdown
    assert "does not claim" in html


def test_run_summary_defaults_to_latest_completed_run(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    config_path = _temp_config(tmp_path)
    first = background_run_once(
        db_path=db_path,
        target_id="SYNTH-TESS-WEAK-001",
        export_reports=False,
        config_path=config_path,
    )
    second = background_run_once(db_path=db_path, export_reports=False, config_path=config_path)

    store = BackgroundStore(db_path)
    latest = store.run_summary()

    assert first.run_id != second.run_id
    assert latest["run"]["run_id"] == second.run_id
    assert latest["needs_follow_up_outcome"]


def test_target_history_and_scheduler_notification(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    config_path = _temp_config(tmp_path)
    result = background_run_once(db_path=db_path, export_reports=False, config_path=config_path)

    store = BackgroundStore(db_path)
    history = store.target_history(result.target_id)
    notification = store.scheduler_notification_summary(result.run_id)

    assert history["runs"]
    assert notification["alert"] is True
    assert notification["outcome"] == "needs_follow_up"


def test_integrity_detects_orphan_rows_and_invalid_reason_codes(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    store = BackgroundStore(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO reviewed_outcomes (
                run_id, target_id, reason_codes_json, negative_evidence_json, summary, created_at
            )
            VALUES ('missing-run', 'TIC X', '["not_a_known_code"]', '[]', 'bad', 'now')
            """
        )

    summary = store.integrity_summary()

    assert summary["ok"] is False
    assert summary["orphan_counts"]["reviewed_outcomes"] == 1
    assert summary["invalid_reason_codes"]["reviewed_outcomes.reason_codes"] == [
        "not_a_known_code"
    ]


def test_lock_wait_records_blocked_run(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    config_path = _temp_config(tmp_path, lock_wait_seconds=0.01)
    store = BackgroundStore(db_path)
    assert store.acquire_run_lock("background_search", "test-owner", 0.0, 0.01, "now")

    result = background_run_once(db_path=db_path, config_path=config_path)

    assert result.outcome.value == "blocked"
    assert result.reason_codes == [ReasonCode.RUN_LOCK_UNAVAILABLE.value]
    assert store.validation_summary()["exactly_one_outcome_per_run"] is True


def test_cli_scheduler_exit_codes_for_needs_follow_up(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    config_path = _temp_config(tmp_path)

    exit_code = main(
        [
            "background-run-once",
            "--db-path",
            str(db_path),
            "--config-path",
            str(config_path),
            "--no-report-export",
            "--scheduler-exit-codes",
        ]
    )

    assert exit_code == EXIT_NEEDS_FOLLOW_UP


def test_v1_schema_migrates_to_v2(tmp_path: Path) -> None:
    db_path = tmp_path / "logs" / "background_search.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
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
                config_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")

    store = BackgroundStore(db_path)
    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(run_ledger)")}
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert "config_version" in columns
    assert "config_fingerprint" in columns
    assert version >= 2
    assert store.validation_summary()["sqlite_schema_version"] >= 2


def _temp_config(tmp_path: Path, lock_wait_seconds: float | None = None) -> Path:
    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["reports"]["export_dir"] = str(tmp_path / "reports" / "background")
    if lock_wait_seconds is not None:
        payload["scheduler"]["lock_wait_seconds"] = lock_wait_seconds
    config_path = tmp_path / "configs" / "background_search_v0.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path
