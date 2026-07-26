"""Behavioral validity and provenance checks for EXO-Hunter."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exo_toolkit.hunter_cross_project import (
    build_sibling_history_manifest,
    load_cross_project_history,
)
from exo_toolkit.hunter_history import load_verified_history_manifest
from exo_toolkit.hunter_models import (
    ArtifactIdentity,
    DecisionValidity,
    ExecutionProvenance,
)
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_SELECTOR_VERSION,
    NEW_SELECTOR_VERSION,
    OPERATOR_SELECTOR_VERSION,
    selection_contract,
)
from exo_toolkit.search_lifecycle import HunterCandidate, HunterStore, TargetExecutionResult


def _candidate() -> HunterCandidate:
    observed = datetime(2026, 1, 1, tzinfo=UTC)
    return HunterCandidate(
        target_id="TIC 1",
        canonical_id="TIC 1",
        source="validity fixture",
        source_provenance={
            "search_category": "new",
            "selector_version": NEW_SELECTOR_VERSION,
        },
        decision_validity=DecisionValidity(
            state="valid",
            source="validity fixture",
            source_version="fixture-v1",
            as_of=observed,
            retrieved_at=observed,
            assessed_at=observed,
            basis="production selector validity fixture",
        ),
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


def test_production_candidate_validation_is_prewrite_atomic(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    invalid = _candidate().model_copy(update={"decision_validity": None})

    with pytest.raises(ValueError, match="lacks decision_validity"):
        store.create_search(
            [invalid],
            requested_target_count=1,
            mode="new",
            selector_version=NEW_SELECTOR_VERSION,
            config={"selection_contract": selection_contract("new")},
        )

    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_manifests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM candidate_catalog").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM search_state_events").fetchone()[0] == 0


def test_cross_project_history_is_durable_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "scan.ndjson"
    source.write_text('{"real":"history"}\n', encoding="utf-8")
    export = tmp_path / "history.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "search_id": "techno-1",
                        "source_project": "2026 Technosignatures",
                        "source_path": source.name,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "entries": [
                            {
                                "target_id": "HIP74981",
                                "canonical_id": "HIP 74981",
                                "searched_at": "2026-01-01T00:00:00+00:00",
                                "status": "searched",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = HunterStore(tmp_path / "hunter.sqlite3")
    summary = store.import_cross_project_history(export, source_root=tmp_path)

    assert summary["validity_state"] == "valid"
    assert summary["source_hashes_verified"] == 1
    assert store.cross_project_searched_identities() == frozenset({"HIP 74981"})
    with (
        store.connect() as connection,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        connection.execute("DELETE FROM cross_project_search_history")


def test_live_sibling_adapter_normalizes_and_verifies_raw_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "results" / "scan_history.ndjson"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "schema_version": "prod_scan_history_v1",
                        "target_stem": "GBT_HIP74981_2026-01-01_ABACAD",
                        "scanned_at_utc": "2026-01-01T00:00:00+00:00",
                        "pathway": "human_review_queue",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "prod_scan_history_v1",
                        "target_stem": "non_catalog_calibrator",
                        "scanned_at_utc": "2026-01-01T01:00:00+00:00",
                        "pathway": "no_signal",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exo_toolkit.hunter_cross_project.sibling_history_source_path",
        lambda _project: source,
    )

    manifest = build_sibling_history_manifest("technosignatures")
    loaded = load_cross_project_history(manifest, source_root=tmp_path)

    assert loaded["validity_state"] == "valid"
    assert loaded["source_hashes_verified"] == 1
    assert len(loaded["entries"]) == 1
    assert loaded["entries"][0]["identities"] == ["HIP 74981"]


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


def test_superseded_v5_snapshot_is_not_current_schema_acceptance() -> None:
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
    assert evidence["ok"] is False
    assert evidence["compressed_snapshot_sha256"] == (
        acceptance["production_snapshot"]["compressed_sha256"]
    )
    assert any("schema version mismatch" in issue for issue in evidence["issues"])
    assert any("cross_project_search_history" in issue for issue in evidence["issues"])


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


def test_v7_acceptance_contracts_remain_historically_frozen() -> None:
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
    assert acceptance["selector_contracts"]["new"]["selector_version"] == (
        "exo_hunter_tic_v2"
    )
    assert acceptance["selector_contracts"]["follow-up"]["selector_version"] == (
        "exo_hunter_follow_up_v3"
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


def test_v12_acceptance_matches_current_prod_closure_contracts() -> None:
    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v12.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["repository_version"] == "0.4.0"
    assert acceptance["selector_contracts"]["new"] == NEW_SELECTOR_VERSION
    assert acceptance["selector_contracts"]["follow_up"] == FOLLOW_UP_SELECTOR_VERSION
    assert acceptance["live_new_workflow"]["targets_failed"] == 0
    assert acceptance["live_new_workflow"]["database_integrity"] == "ok"
    assert acceptance["live_follow_up_workflow"]["targets_failed"] == 0
    assert (
        acceptance["live_follow_up_workflow"][
            "weak_quality_did_not_block_best_available_selection"
        ]
        is True
    )
    assert acceptance["adaptive_outside_initial_sample"][
        "high_value_target_beyond_initial_page_or_retained_pool_selected"
    ]
    assert acceptance["validity_and_atomicity"]["candidate_file_cli_present"] is False
    assert acceptance["cross_project_history"]["source_hashes_verified_live"] == 1


def test_hunter_operator_docs_do_not_advertise_retired_bypasses() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = Path("docs/HUNTER_PRODUCTION_WORKFLOW.md").read_text(
        encoding="utf-8"
    )

    assert "can also consume a provenance-complete JSON or CSV candidate" not in readme
    assert "A candidate file may be supplied to `Create-New-Search`" not in workflow
    assert "does not accept operator-ranked candidate files" in readme
    assert "strict read-only `prod_scan_history_v1` adapter" in workflow
