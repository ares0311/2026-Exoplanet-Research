"""Behavioral validity and provenance checks for EXO-Hunter."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exo_toolkit.hunter_history import load_verified_history_manifest
from exo_toolkit.hunter_models import ArtifactIdentity, ExecutionProvenance
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_SELECTOR_VERSION,
    NEW_SELECTOR_VERSION,
    OPERATOR_SELECTOR_VERSION,
    selection_contract,
)
from exo_toolkit.search_lifecycle import HunterCandidate, HunterStore, TargetExecutionResult


def _candidate() -> HunterCandidate:
    return HunterCandidate(
        target_id="TIC 1",
        canonical_id="TIC 1",
        source="validity fixture",
        source_provenance={
            "search_category": "new",
            "selector_version": NEW_SELECTOR_VERSION,
        },
        ranking_score=1.0,
        selection_reason="deterministic validity fixture",
        metrics={"expected_information_gain": 0.5, "scientific_suitability": 0.5},
    )


def _search(store: HunterStore) -> dict[str, object]:
    return store.create_search(
        [_candidate()],
        requested_target_count=1,
        mode="new",
        selector_version="validity-test-v1",
        config={"seed": 1},
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_history_import_rejects_source_hash_drift_before_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("original", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "search_id": "history-1",
                "mode": "new",
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T01:00:00+00:00",
                "searched_by": "EXO-Hunter",
                "source_project": "test",
                "method_or_data": "fixture",
                "source_path": str(source),
                "source_sha256": "0" * 64,
                "provenance_uri": "fixture:source",
                "entries": [],
            }
        ],
    }
    store = HunterStore(tmp_path / "hunter.sqlite3")

    with pytest.raises(ValueError, match="source hash mismatch"):
        store.import_history_manifest(manifest)

    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_manifests").fetchone()[0] == 0


def test_sqlite_triggers_enforce_append_only_history(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _search(store)
    store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=str(search["search_id"]),
    )

    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM target_search_history")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE search_manifests SET mode='follow-up'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM candidate_catalog")


def test_validity_detects_candidate_content_tampering(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    _search(store)
    with store.connect() as connection:
        connection.execute("DROP TRIGGER hunter_immutable_candidate_catalog_update")
        connection.execute("UPDATE candidate_catalog SET candidate_json='{}'")

    summary = store.validity_summary()
    assert summary["ok"] is False
    assert any("invalid candidate" in issue for issue in summary["issues"])
    assert any("missing immutable-history triggers" in issue for issue in summary["issues"])


def test_validity_detects_selector_contract_drift(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    store.create_search(
        [_candidate()],
        requested_target_count=1,
        mode="new",
        selector_version=NEW_SELECTOR_VERSION,
        config={"selection_contract": selection_contract("new")},
    )
    assert store.validity_summary()["ok"] is True
    with store.connect() as connection:
        connection.execute("DROP TRIGGER hunter_immutable_search_manifests_update")
        connection.execute(
            "UPDATE search_manifests SET config_json=?",
            ('{"selection_contract":{"selector_version":"wrong"}}',),
        )
    summary = store.validity_summary()
    assert summary["ok"] is False
    assert any("selection contract does not match" in issue for issue in summary["issues"])


def test_validity_detects_model_artifact_drift(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    model.write_text("model-v1", encoding="utf-8")
    artifact = ArtifactIdentity(
        role="xgboost_model",
        path=str(model),
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        size_bytes=model.stat().st_size,
    )
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _search(store)
    provenance = ExecutionProvenance(
        candidate_snapshot=_candidate(),
        pipeline_context={"raw_uris": ["fixture:lightcurve"]},
        code_version="test",
        git_commit="a" * 40,
        scorer="xgboost",
        model_artifacts=(artifact,),
        runner="fixture",
    )
    store.execute_search(
        lambda _: TargetExecutionResult(
            status="no_signal", result={"interpretation": "null"}, provenance=provenance
        ),
        search_id=str(search["search_id"]),
    )
    assert store.validity_summary()["ok"] is True

    model.write_text("tampered", encoding="utf-8")
    summary = store.validity_summary()
    assert summary["ok"] is False
    assert any("model artifact hash mismatch" in issue for issue in summary["issues"])


def test_committed_history_sources_are_verified_byte_for_byte() -> None:
    manifest = Path("data_selection/hunter_prior_search_history_v1.json")
    payload = load_verified_history_manifest(manifest)
    sources = payload["sources"]
    events = [entry for source in sources for entry in source["entries"]]
    assert len(sources) == 7
    assert len(events) == 608
    assert len({entry["target_id"] for entry in events}) == 200
    for source in sources:
        source_path = Path(str(source["source_path"]))
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["source_sha256"]


def test_acceptance_validator_returns_nonzero_for_invalid_database(tmp_path: Path) -> None:
    from Skills.validate_hunter_acceptance import main

    db = tmp_path / "hunter.sqlite3"
    store = HunterStore(db)
    _search(store)
    with store.connect() as connection:
        connection.execute("DROP TRIGGER hunter_immutable_search_manifests_delete")

    assert main(["--db", str(db)]) == 2


def test_committed_acceptance_snapshot_matches_structured_evidence() -> None:
    from Skills.validate_hunter_acceptance import _validate

    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v5.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = Path(acceptance["production_snapshot"]["path"])
    evidence = _validate(
        snapshot,
        Path("data_selection/hunter_prior_search_history_v1.json"),
        None,
    )
    assert evidence["ok"] is True
    assert evidence["issues"] == []
    for field in (
        "compressed_snapshot_sha256",
        "database_sha256",
        "schema_version",
        "sqlite_integrity",
        "foreign_key_violation_count",
        "immutable_trigger_count",
        "manifest_hashes_verified",
        "snapshot_hashes_verified",
        "provenance_rows_verified",
        "counts",
        "history_status_counts",
    ):
        expected_field = field.removeprefix("compressed_snapshot_")
        if field == "compressed_snapshot_sha256":
            expected_field = "compressed_sha256"
        assert evidence[field] == acceptance["production_snapshot"][expected_field]
    assert all(
        requirement["verified"] is True
        for requirement in acceptance["requirements"].values()
    )


def test_v6_acceptance_selector_contracts_remain_historically_frozen() -> None:
    """v6 is superseded (see v7); its own frozen content must not silently drift."""
    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v6.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["selector_contracts"]["follow-up"]["selector_version"] == (
        "exo_hunter_follow_up_v2"
    )
    assert acceptance["selector_contracts"]["follow-up"]["fpp_max_exclusive"] == 0.15


def test_current_acceptance_matches_executable_selector_contracts() -> None:
    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v7.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = json.loads(
        Path(acceptance["baseline_acceptance"]["artifact"]).read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["baseline_acceptance"]["acceptance_id"] == baseline["acceptance_id"]
    assert (
        acceptance["baseline_acceptance"]["compressed_snapshot_sha256"]
        == baseline["production_snapshot"]["compressed_sha256"]
    )
    assert acceptance["selector_contracts"]["new"] == selection_contract("new")
    assert acceptance["selector_contracts"]["follow-up"] == selection_contract("follow-up")
    assert acceptance["selector_contracts"]["operator"] == selection_contract(
        "new", operator_supplied=True
    )
    assert acceptance["selector_contracts"]["new"]["selector_version"] == (
        NEW_SELECTOR_VERSION
    )
    assert acceptance["selector_contracts"]["follow-up"]["selector_version"] == (
        FOLLOW_UP_SELECTOR_VERSION
    )
    assert acceptance["selector_contracts"]["operator"]["selector_version"] == (
        OPERATOR_SELECTOR_VERSION
    )
    assert acceptance["live_business_validation"]["create_search_after_fix"] == {
        "mode": "follow-up",
        "requested_target_count": 10,
        "selected_target_count": 10,
        "candidate_pool_count": 202,
        "shortfall": 0,
        "database_integrity_before": True,
        "database_integrity_after": True,
    }
