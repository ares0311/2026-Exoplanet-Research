"""Tests for the durable EXO-Hunter search lifecycle."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from exo_toolkit.search_lifecycle import (
    FollowUpRecommendation,
    HunterCandidate,
    HunterStore,
    PriorSearch,
    TargetExecutionResult,
    format_eta,
)


def _candidate(index: int, *, score: float | None = None) -> HunterCandidate:
    return HunterCandidate(
        target_id=f"TIC {index}",
        canonical_id=f"TIC {index}",
        aliases=(str(index),),
        source="test catalog",
        source_provenance={"search_category": "new", "row": index},
        estimated_download_gb=0.001,
        ranking_score=float(index if score is None else score),
        selection_reason="deterministic test rank",
        metrics={"priority": float(index)},
    )


def test_format_eta_never_emits_sixty_second_remainder() -> None:
    assert format_eta(239.6) == "4m00s"
    assert format_eta(89.6) == "90s"


def _prior(index: int = 1) -> PriorSearch:
    return PriorSearch(
        searched_by="external researcher",
        searched_at=datetime(2025, 1, 1, tzinfo=UTC),
        source_project="external survey",
        method_or_data="published TESS light curve search",
        result=f"candidate {index} requires follow-up",
        provenance_uri=f"doi:test/{index}",
    )


def _create(
    store: HunterStore,
    candidates: list[HunterCandidate],
    *,
    count: int,
    mode: str = "new",
) -> dict[str, Any]:
    return store.create_search(
        candidates,
        requested_target_count=count,
        mode=mode,  # type: ignore[arg-type]
        selector_version="test-v1",
        config={"seed": 42},
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_selects_100_deterministically_from_10000_candidate_universe(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(i) for i in range(10_000)], count=100)

    assert search["candidate_pool_count"] == 10_000
    assert search["selected_target_count"] == 100
    assert [row["target_id"] for row in search["targets"][:3]] == [
        "TIC 9999",
        "TIC 9998",
        "TIC 9997",
    ]
    assert search["state"] == "pending"
    assert len(search["manifest_sha256"]) == 64


def test_ineligible_candidates_never_enter_manifest(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    rejected = _candidate(99).model_copy(
        update={"eligible": False, "eligibility_reason": "known variable"}
    )
    search = _create(store, [rejected, _candidate(1)], count=1)
    assert search["targets"][0]["target_id"] == "TIC 1"


def test_insufficient_eligible_pool_fails_without_creating_search(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    with pytest.raises(RuntimeError, match="only 1 eligible"):
        _create(store, [_candidate(1)], count=2)
    assert store.open_searches() == []


def test_second_pending_search_is_refused(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    first = _create(store, [_candidate(1)], count=1)
    with pytest.raises(RuntimeError, match=first["search_id"]):
        store.create_search(
            [_candidate(2)],
            requested_target_count=1,
            mode="new",
            selector_version="test-v1",
            config={},
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_partial_run_is_loud_and_resume_retries_only_failure(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1), _candidate(2), _candidate(3)], count=3)
    calls: list[str] = []
    fail_once = {"TIC 2"}

    def runner(candidate: HunterCandidate) -> TargetExecutionResult:
        calls.append(candidate.target_id)
        if candidate.target_id in fail_once:
            fail_once.remove(candidate.target_id)
            raise RuntimeError("bounded fetch failed")
        return TargetExecutionResult(
            status="no_signal",
            result={"composite_interpretation": "no signal"},
            provenance={"raw_uris": [f"mast:{candidate.target_id}"]},
        )

    first = store.execute_search(runner, search_id=search["search_id"], workers=3)
    assert first.status == "partial"
    assert first.targets_succeeded == 2
    assert first.targets_failed == 1
    assert store.current_state(search["search_id"]) == "partial"

    calls.clear()
    second = store.execute_search(runner, search_id=search["search_id"], workers=3)
    assert second.status == "completed"
    assert calls == ["TIC 2"]
    assert second.targets_already_complete == 2
    assert len(store.target_history("TIC 2")) == 2


def test_results_and_provenance_are_append_only(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)

    def fail(candidate: HunterCandidate) -> TargetExecutionResult:
        return TargetExecutionResult(
            status="failed",
            result={"candidate": candidate.target_id},
            provenance={"attempt": 1},
            error_message="network timeout",
        )

    def succeed(candidate: HunterCandidate) -> TargetExecutionResult:
        return TargetExecutionResult(
            status="no_signal",
            result={"candidate": candidate.target_id, "interpretation": "null"},
            provenance={"attempt": 2},
        )

    assert store.execute_search(fail, search_id=search["search_id"]).status == "failed"
    assert store.execute_search(succeed, search_id=search["search_id"]).status == "completed"
    history = store.target_history("TIC 1")
    assert [row["status"] for row in history] == ["failed", "no_signal"]
    assert [row["provenance"]["attempt"] for row in history] == [1, 2]


def test_new_mode_excludes_targets_with_terminal_history(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=search["search_id"],
    )
    next_search = store.create_search(
        [_candidate(1, score=100), _candidate(2, score=1)],
        requested_target_count=1,
        mode="new",
        selector_version="test-v1",
        config={},
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert next_search["targets"][0]["target_id"] == "TIC 2"


def test_candidate_result_registers_actionable_follow_up(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    outcome = TargetExecutionResult(
        status="candidate_found",
        result={"composite_interpretation": "candidate signal"},
        provenance={"raw_uris": ["mast:product"]},
        follow_ups=(
            FollowUpRecommendation(
                candidate_id="TIC1-s01",
                priority=95.0,
                reason="low FPP with independent events",
                evidence={"fpp": 0.05},
                recommended_action="obtain centroid evidence",
            ),
        ),
    )
    summary = store.execute_search(lambda _: outcome, search_id=search["search_id"])
    follow_ups = store.list_follow_ups()

    assert summary.follow_ups_registered == 1
    assert follow_ups[0]["target_id"] == "TIC 1"
    assert follow_ups[0]["evidence"] == {"fpp": 0.05}
    assert follow_ups[0]["recommended_action"] == "obtain centroid evidence"


def test_follow_up_mode_requires_reliable_prior_search_provenance(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    without_history = _candidate(1).model_copy(
        update={"source_provenance": {"search_category": "follow-up"}}
    )
    with pytest.raises(ValidationError, match="prior_searches"):
        HunterCandidate.model_validate(without_history.model_dump())

    follow_up = _candidate(2).model_copy(
        update={
            "source_provenance": {"search_category": "follow-up"},
            "prior_searches": (_prior(2),),
        }
    )
    search = _create(store, [follow_up], count=1, mode="follow-up")
    restored = HunterCandidate.model_validate(search["targets"][0]["candidate"])
    assert restored.prior_searches[0].source_project == "external survey"


def test_follow_up_registry_can_seed_later_follow_up_search(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    store.execute_search(
        lambda _: TargetExecutionResult(
            status="candidate_found",
            result={},
            provenance={},
            follow_ups=(
                FollowUpRecommendation(
                    candidate_id="TIC1-s01",
                    priority=99.0,
                    reason="unresolved signal",
                    evidence={"fpp": 0.01},
                    recommended_action="repeat photometry",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    candidates = store.follow_up_candidates()
    follow_search = store.create_search(
        candidates,
        requested_target_count=1,
        mode="follow-up",
        selector_version="test-v1",
        config={},
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert follow_search["targets"][0]["target_id"] == "TIC 1"
    assert follow_search["targets"][0]["candidate"]["prior_searches"]


def test_deferred_follow_up_is_visible_but_not_search_eligible(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    store.execute_search(
        lambda _: TargetExecutionResult(
            status="candidate_found",
            result={},
            provenance={},
            follow_ups=(
                FollowUpRecommendation(
                    candidate_id="TIC1-s01",
                    priority=99.0,
                    reason="evidence limited",
                    evidence={"events": 2},
                    recommended_action="wait for more observations",
                    search_eligible=False,
                    revisit_reason="four independent events are required",
                ),
            ),
        ),
        search_id=search["search_id"],
    )

    rows = store.list_follow_ups()
    assert rows[0]["search_eligible"] is False
    assert rows[0]["revisit_reason"] == "four independent events are required"
    assert store.follow_up_candidates() == []


def test_reviewed_import_preserves_history_and_is_idempotent(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    candidate = _candidate(7).model_copy(
        update={
            "source_provenance": {"search_category": "follow-up"},
            "prior_searches": (_prior(7),),
        }
    )
    recommendation = FollowUpRecommendation(
        candidate_id="TIC7-s02",
        priority=88.0,
        reason="reviewed signal",
        evidence={"events": 2},
        recommended_action="await another event",
        search_eligible=False,
        revisit_reason="new event-covering observations required",
    )
    kwargs = {
        "candidate": candidate,
        "recommendation": recommendation,
        "source_search_id": "legacy-batch",
        "source_attempt_id": "legacy-shards",
        "source_result": {"review_status": "plausible_but_weak"},
        "source_provenance": {"artifact": "review.json"},
        "completed_at": datetime(2026, 1, 3, tzinfo=UTC),
        "imported_at": datetime(2026, 1, 4, tzinfo=UTC),
    }

    first = store.import_reviewed_follow_up(**kwargs)
    second = store.import_reviewed_follow_up(**kwargs)

    assert first["created"] is True
    assert second == {**first, "created": False}
    assert store.get_search(first["search_id"])["state"] == "completed"
    assert store.open_searches() == []
    assert store.target_history("TIC 7")[0]["result"]["review_status"] == "plausible_but_weak"
    assert len(store.list_follow_ups()) == 1
    assert store.follow_up_candidates() == []


def test_manifest_csv_is_operator_review_artifact_not_system_of_record(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1), _candidate(2)], count=2)
    output = store.export_manifest_csv(search["search_id"], tmp_path / "manifest.csv")
    text = output.read_text(encoding="utf-8")
    assert "target_id" in text
    assert "TIC 2" in text
    assert store.get_search(search["search_id"])["targets"]


def test_integrity_reports_all_required_durable_concepts(tmp_path: Path) -> None:
    db_path = tmp_path / "hunter.sqlite3"
    store = HunterStore(db_path)
    assert store.integrity_summary() == {
        "ok": True,
        "sqlite_integrity": "ok",
        "missing_tables": [],
        "orphan_manifest_targets": 0,
        "schema_version": 2,
    }
    connection = sqlite3.connect(db_path)
    names = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    connection.close()
    assert {
        "candidate_catalog",
        "search_manifests",
        "search_runs",
        "target_search_history",
        "follow_up_registry",
    } <= names


def test_schema_v1_follow_up_table_migrates_without_losing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "hunter.sqlite3"
    store = HunterStore(db_path)
    search = _create(store, [_candidate(1)], count=1)
    store.execute_search(
        lambda _: TargetExecutionResult(
            status="candidate_found",
            result={},
            provenance={},
            follow_ups=(
                FollowUpRecommendation(
                    candidate_id="signal",
                    priority=90,
                    reason="signal",
                    evidence={},
                    recommended_action="review",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE follow_up_registry RENAME TO follow_up_registry_v2")
    connection.execute(
        "CREATE TABLE follow_up_registry AS SELECT follow_up_id, search_id, target_id, "
        "candidate_id, priority, reason, evidence_json, prior_search_provenance_json, "
        "recommended_action, status, created_at FROM follow_up_registry_v2"
    )
    connection.execute("DROP TABLE follow_up_registry_v2")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    migrated = HunterStore(db_path)
    row = migrated.list_follow_ups()[0]
    assert row["candidate_id"] == "signal"
    assert row["search_eligible"] is True
    assert row["revisit_reason"] is None


def test_execute_refuses_completed_search(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=search["search_id"],
    )
    with pytest.raises(RuntimeError, match="not executable"):
        store.execute_search(
            lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
            search_id=search["search_id"],
        )


def test_attempt_timestamps_and_counts_are_consistent(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=1)
    times = iter(
        [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=2),
        ]
    )
    summary = store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=search["search_id"],
        now_fn=lambda: next(times),
    )
    assert summary.targets_processed == 1
    assert summary.targets_succeeded == 1
    assert summary.completed_at > summary.started_at


def test_restart_marks_abandoned_attempt_interrupted_before_resume(tmp_path: Path) -> None:
    db_path = tmp_path / "hunter.sqlite3"
    store = HunterStore(db_path)
    search = _create(store, [_candidate(1)], count=1)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO search_runs "
        "(attempt_id, search_id, status, started_at, config_json) "
        "VALUES ('abandoned', ?, 'running', '2026-01-01T00:00:00+00:00', '{}')",
        (search["search_id"],),
    )
    connection.execute(
        "INSERT INTO search_state_events "
        "(search_id, state, detail_json, created_at) "
        "VALUES (?, 'running', '{}', '2026-01-01T00:00:00+00:00')",
        (search["search_id"],),
    )
    connection.commit()
    connection.close()

    summary = store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=search["search_id"],
    )
    connection = sqlite3.connect(db_path)
    abandoned = connection.execute(
        "SELECT status, error_message FROM search_runs WHERE attempt_id='abandoned'"
    ).fetchone()
    connection.close()

    assert summary.status == "completed"
    assert abandoned == (
        "interrupted",
        "Previous process ended before finalizing this attempt",
    )
