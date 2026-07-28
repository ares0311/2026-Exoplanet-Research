"""Tests for the durable EXO-Hunter search lifecycle."""
from __future__ import annotations

import csv
import hashlib
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


def _history_manifest(
    tmp_path: Path, *, include_follow_up: bool = True
) -> dict[str, Any]:
    prior_source = tmp_path / "prior.json"
    prior_source.write_text("historical new search", encoding="utf-8")
    follow_up_source = tmp_path / "prior-follow-up.json"
    follow_up_source.write_text("historical follow-up search", encoding="utf-8")
    source = {
        "search_id": "historical-new-search",
        "mode": "new",
        "started_at": "2025-01-01T00:00:00+00:00",
        "completed_at": "2025-01-01T01:00:00+00:00",
        "searched_by": "EXO-Hunter",
        "source_project": "legacy project",
        "method_or_data": "TESS QLP",
        "source_path": str(prior_source),
        "source_sha256": hashlib.sha256(prior_source.read_bytes()).hexdigest(),
        "provenance_uri": "artifact:prior-new",
        "entries": [
            {
                "target_id": "TIC 1",
                "status": "candidate_found",
                "searched_at": "2025-01-01T00:30:00+00:00",
                "ranking_score": 0.9,
                "metrics": {"best_fpp": 0.1},
                "result": {"interpretation": "promising"},
                "best_fpp": 0.1,
                "best_detection_confidence": 0.8,
                "best_pathway": "planet_hunters_discussion",
                "error_message": None,
            },
            {
                "target_id": "TIC 2",
                "status": "no_signal",
                "searched_at": "2025-01-01T00:45:00+00:00",
                "ranking_score": 0.5,
                "metrics": {},
                "result": {"interpretation": "clear"},
                "best_fpp": None,
                "best_pathway": None,
                "error_message": None,
            },
        ],
    }
    sources = [source]
    if include_follow_up:
        sources.append(
            {
                **source,
                "search_id": "historical-follow-up-search",
                "mode": "follow-up",
                "started_at": "2025-02-01T00:00:00+00:00",
                "completed_at": "2025-02-01T01:00:00+00:00",
                "source_path": str(follow_up_source),
                "source_sha256": hashlib.sha256(follow_up_source.read_bytes()).hexdigest(),
                "provenance_uri": "artifact:prior-follow-up",
                "entries": [
                    {
                        **source["entries"][0],
                        "searched_at": "2025-02-01T00:30:00+00:00",
                    }
                ],
            }
        )
    return {"schema_version": 1, "sources": sources}


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


def test_history_manifest_import_is_idempotent_and_preserves_repeated_work(
    tmp_path: Path,
) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    manifest = _history_manifest(tmp_path)

    assert store.import_history_manifest(manifest) == {
        "sources_total": 2,
        "sources_created": 2,
        "events_created": 3,
    }
    assert store.import_history_manifest(manifest) == {
        "sources_total": 2,
        "sources_created": 0,
        "events_created": 0,
    }
    universe = {row.target_id: row for row in store.follow_up_universe()}
    assert set(universe) == {"TIC 1", "TIC 2"}
    assert len(universe["TIC 1"].prior_searches) == 2
    assert universe["TIC 1"].eligible is False
    assert universe["TIC 1"].eligibility_reason == (
        "latest durable search already performed follow-up"
    )
    assert len(store.target_history("TIC 1")) == 2


def test_historical_partial_run_is_visible_but_not_resumable(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    manifest = _history_manifest(tmp_path, include_follow_up=False)
    failed_entry = manifest["sources"][0]["entries"][1]
    failed_entry["status"] = "failed"
    failed_entry["error_message"] = "historical fetch failed"

    store.import_history_manifest(manifest)

    assert store.current_state("historical-new-search") == "archived_partial"
    assert store.open_searches() == []
    history = store.target_history("TIC 2")
    assert history[0]["status"] == "failed"
    assert history[0]["error_message"] == "historical fetch failed"


def test_history_universe_selects_unfollowed_qualifying_new_result(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    store.import_history_manifest(_history_manifest(tmp_path, include_follow_up=False))

    universe = store.follow_up_universe()
    assert len(universe) == 2
    assert universe[0].target_id == "TIC 1"
    assert universe[0].eligible is True
    search = _create(store, universe, count=1, mode="follow-up")
    assert search["candidate_pool_count"] == 2
    assert search["targets"][0]["target_id"] == "TIC 1"
    assert len(search["targets"][0]["candidate"]["prior_searches"]) == 1


def test_follow_up_universe_includes_signals_below_strict_production_bar(
    tmp_path: Path,
) -> None:
    """A real candidate_found detection that misses the strict FPP/confidence bar
    must remain a selectable (lower-ranked) follow-up candidate, not disappear
    entirely (directive: do not fail a normal top-N request because no candidate
    crosses an arbitrary quality threshold)."""
    store = HunterStore(tmp_path / "hunter.sqlite3")
    manifest = _history_manifest(tmp_path, include_follow_up=False)
    weak_entry = manifest["sources"][0]["entries"][0]
    weak_entry["best_fpp"] = 0.42
    weak_entry["best_detection_confidence"] = 0.55
    weak_entry["best_pathway"] = "planet_hunters_discussion"
    store.import_history_manifest(manifest)

    universe = {row.target_id: row for row in store.follow_up_universe()}
    weak = universe["TIC 1"]
    assert weak.eligible is True
    # HunterCandidate.metrics is float|int|str|None typed, so bool coerces to 0.0/1.0.
    assert bool(weak.metrics["meets_strict_follow_up_bar"]) is False
    assert weak.eligibility_reason == (
        "new-search evidence found a signal below the strict follow-up bar"
    )

    search = _create(store, list(universe.values()), count=1, mode="follow-up")
    assert search["targets"][0]["target_id"] == "TIC 1"


def test_ineligible_candidates_never_enter_manifest(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    rejected = _candidate(99).model_copy(
        update={"eligible": False, "eligibility_reason": "known variable"}
    )
    search = _create(store, [rejected, _candidate(1)], count=1)
    assert search["targets"][0]["target_id"] == "TIC 1"


def test_insufficient_pool_returns_best_available_fewer_than_requested(
    tmp_path: Path,
) -> None:
    """A shortfall must never block a normal top-N request (directive: rank and
    absolute quality are different; return fewer than N only when fewer than N
    valid candidates actually exist)."""
    store = HunterStore(tmp_path / "hunter.sqlite3")
    search = _create(store, [_candidate(1)], count=2)
    assert search["requested_target_count"] == 2
    assert search["selected_target_count"] == 1
    assert search["targets"][0]["target_id"] == "TIC 1"
    assert store.open_searches()[0]["search_id"] == search["search_id"]


def test_zero_available_candidates_still_fails_without_creating_search(
    tmp_path: Path,
) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    rejected = _candidate(1).model_copy(
        update={"eligible": False, "eligibility_reason": "already searched"}
    )
    with pytest.raises(RuntimeError, match="No new candidates are currently available"):
        _create(store, [rejected], count=1)
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
    assert [
        row["provenance"]["pipeline_context"]["runner_provenance"]["attempt"]
        for row in history
    ] == [1, 2]
    assert all(row["provenance"]["schema_version"] == 1 for row in history)


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
    source_follow_up_id = store.list_follow_ups()[0]["follow_up_id"]
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
    assert store.list_follow_ups() == []
    assert store.list_follow_ups(status="scheduled")[0]["follow_up_id"] == source_follow_up_id
    assert store.follow_up_candidates() == []

    store.execute_search(
        lambda _: TargetExecutionResult(
            status="candidate_found",
            result={"interpretation": "signal persists"},
            provenance={"follow_up_search": follow_search["search_id"]},
            follow_ups=(
                FollowUpRecommendation(
                    candidate_id="TIC1-s02",
                    priority=80.0,
                    reason="new evidence requires later observations",
                    evidence={"events": 3},
                    recommended_action="wait for another event",
                    search_eligible=False,
                    revisit_reason="four events are required",
                ),
            ),
        ),
        search_id=follow_search["search_id"],
    )

    source = store.list_follow_ups(status="completed")[0]
    child = store.list_follow_ups(status="deferred")[0]
    assert source["follow_up_id"] == source_follow_up_id
    assert child["parent_follow_up_id"] == source_follow_up_id
    assert [event["state"] for event in source["events"]] == [
        "open",
        "scheduled",
        "completed",
    ]
    assert child["events"][0]["state"] == "deferred"
    assert store.follow_up_candidates() == []
    with pytest.raises(RuntimeError, match="is not open: state=completed"):
        store.create_search(
            candidates,
            requested_target_count=1,
            mode="follow-up",
            selector_version="test-v1",
            config={},
            now=datetime(2026, 1, 3, tzinfo=UTC),
        )
    assert store.open_searches() == []


def test_failed_follow_up_stays_scheduled_until_successful_resume(tmp_path: Path) -> None:
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
                    evidence={},
                    recommended_action="repeat photometry",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    follow_search = _create(store, store.follow_up_candidates(), count=1, mode="follow-up")
    assert store.execute_search(
        lambda _: TargetExecutionResult(
            status="failed",
            result={},
            provenance={},
            error_message="temporary archive failure",
        ),
        search_id=follow_search["search_id"],
    ).status == "failed"
    assert store.list_follow_ups(status="scheduled")

    assert store.execute_search(
        lambda _: TargetExecutionResult(status="no_signal", result={}, provenance={}),
        search_id=follow_search["search_id"],
    ).status == "completed"
    row = store.list_follow_ups(status="completed")[0]
    assert [event["state"] for event in row["events"]] == [
        "open",
        "scheduled",
        "attempt_failed",
        "completed",
    ]


def test_no_data_follow_up_is_deferred_not_rescheduled(tmp_path: Path) -> None:
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
                    evidence={},
                    recommended_action="obtain more data",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    follow_search = _create(store, store.follow_up_candidates(), count=1, mode="follow-up")
    store.execute_search(
        lambda _: TargetExecutionResult(status="no_data", result={}, provenance={}),
        search_id=follow_search["search_id"],
    )
    row = store.list_follow_ups(status="deferred")[0]
    assert row["search_eligible"] is False
    assert "new archive products" in row["revisit_reason"]
    assert row["events"][-1]["detail"]["outcome_status"] == "no_data"
    assert store.follow_up_candidates() == []


def _deferred_follow_up(tmp_path: Path) -> tuple[HunterStore, str]:
    """Build a real no_data-deferred follow-up row and return (store, follow_up_id)."""
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
                    evidence={},
                    recommended_action="obtain more data",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    follow_search = _create(store, store.follow_up_candidates(), count=1, mode="follow-up")
    store.execute_search(
        lambda _: TargetExecutionResult(status="no_data", result={}, provenance={}),
        search_id=follow_search["search_id"],
    )
    follow_up_id = store.list_follow_ups(status="deferred")[0]["follow_up_id"]
    return store, str(follow_up_id)


class TestRecordSectorRecheck:
    def test_first_recheck_with_any_sectors_flips_to_open(self, tmp_path: Path) -> None:
        store, follow_up_id = _deferred_follow_up(tmp_path)

        result = store.record_sector_recheck(
            follow_up_id, sectors=[42], checked_at=datetime.now(UTC)
        )

        assert result["grew"] is True
        assert result["new_sectors"] == [42]
        assert store.list_follow_ups(status="open")
        assert store.list_follow_ups(status="deferred") == []

    def test_recheck_with_zero_sectors_stays_deferred(self, tmp_path: Path) -> None:
        store, follow_up_id = _deferred_follow_up(tmp_path)

        result = store.record_sector_recheck(
            follow_up_id, sectors=[], checked_at=datetime.now(UTC)
        )

        assert result["grew"] is False
        assert store.list_follow_ups(status="deferred")
        assert store.list_follow_ups(status="open") == []

    def test_recheck_always_advances_baseline_and_timestamp(self, tmp_path: Path) -> None:
        store, follow_up_id = _deferred_follow_up(tmp_path)
        checked_at = datetime(2026, 7, 24, tzinfo=UTC)

        store.record_sector_recheck(follow_up_id, sectors=[], checked_at=checked_at)

        row = store.list_follow_ups(status="deferred")[0]
        assert row["last_known_sectors"] == "[]"
        assert row["last_mast_checked_at"] == checked_at.isoformat()

    def test_second_recheck_only_flips_on_real_growth_since_last_check(
        self, tmp_path: Path
    ) -> None:
        # Simulate a deferred row that already has a recorded baseline from an
        # earlier recheck that did not grow (a fresh row's baseline is always
        # empty, so growth from a truly empty baseline flips immediately --
        # this proves the comparison is against the *last recorded* baseline,
        # not always-empty original deferral state).
        store, follow_up_id = _deferred_follow_up(tmp_path)
        with store.connect() as connection:
            connection.execute(
                "UPDATE follow_up_registry SET last_known_sectors=? WHERE follow_up_id=?",
                ("[10, 11]", follow_up_id),
            )

        same_result = store.record_sector_recheck(
            follow_up_id, sectors=[10, 11], checked_at=datetime.now(UTC)
        )
        assert same_result["grew"] is False
        assert store.list_follow_ups(status="deferred")

        grown_result = store.record_sector_recheck(
            follow_up_id, sectors=[10, 11, 12], checked_at=datetime.now(UTC)
        )
        assert grown_result["grew"] is True
        assert grown_result["new_sectors"] == [12]
        assert store.list_follow_ups(status="open")

    def test_recheck_appends_event_log_entry_even_when_not_grown(
        self, tmp_path: Path
    ) -> None:
        store, follow_up_id = _deferred_follow_up(tmp_path)

        store.record_sector_recheck(follow_up_id, sectors=[], checked_at=datetime.now(UTC))

        events = store.follow_up_events(follow_up_id)
        assert events[-1]["state"] == "deferred"
        assert events[-1]["detail"]["recheck"] is True
        assert events[-1]["detail"]["grew"] is False

    def test_recheck_appends_open_transition_event_when_grown(
        self, tmp_path: Path
    ) -> None:
        store, follow_up_id = _deferred_follow_up(tmp_path)

        store.record_sector_recheck(follow_up_id, sectors=[5], checked_at=datetime.now(UTC))

        events = store.follow_up_events(follow_up_id)
        assert events[-1]["state"] == "open"
        assert events[-1]["detail"]["recheck"] is True
        assert events[-1]["detail"]["new_sectors"] == [5]

    def test_unknown_follow_up_id_raises(self, tmp_path: Path) -> None:
        store = HunterStore(tmp_path / "hunter.sqlite3")
        with pytest.raises(RuntimeError, match="Unknown follow-up registry row"):
            store.record_sector_recheck(
                "does-not-exist", sectors=[1], checked_at=datetime.now(UTC)
            )

    def test_non_deferred_follow_up_raises(self, tmp_path: Path) -> None:
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
                        evidence={},
                        recommended_action="obtain more data",
                    ),
                ),
            ),
            search_id=search["search_id"],
        )
        open_follow_up_id = store.list_follow_ups(status="open")[0]["follow_up_id"]
        with pytest.raises(RuntimeError, match="is not deferred"):
            store.record_sector_recheck(
                open_follow_up_id, sectors=[1], checked_at=datetime.now(UTC)
            )


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

    assert store.list_follow_ups() == []
    rows = store.list_follow_ups(status="deferred")
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
    assert store.list_follow_ups() == []
    assert len(store.list_follow_ups(status="deferred")) == 1
    assert store.follow_up_candidates() == []


def test_manifest_csv_is_operator_review_artifact_not_system_of_record(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    observed = datetime(2025, 1, 1, tzinfo=UTC)
    candidate = _candidate(1).model_copy(
        update={
            "distance_pc": 10.0,
            "metrics": {
                "tmag": 12.3,
                "teff_k": 5200,
                "radius_rsun": 0.9,
                "qlp_product_count": 3,
                "expected_information_gain": 0.5,
                "latest_fpp": 0.1,
                "latest_detection_confidence": 0.8,
                "latest_pathway": "planet_hunters_discussion",
                "meets_strict_follow_up_bar": True,
            },
            "prior_searches": (
                PriorSearch(
                    searched_by="External Researcher",
                    searched_at=observed,
                    source_project="External Survey",
                    method_or_data="TESS photometry",
                    result="unresolved signal",
                    provenance_uri="doi:example",
                ),
            ),
        }
    )
    search = _create(store, [candidate, _candidate(2)], count=2)
    exported_at = datetime(2026, 1, 2, tzinfo=UTC)
    output = store.export_manifest_csv(
        search["search_id"],
        tmp_path / "manifest.csv",
        exported_at=exported_at,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["target_id"] == "TIC 1")

    assert float(row["distance_light_years"]) == pytest.approx(32.61563777)
    assert row["search_status"] == "new"
    assert row["prior_search_count"] == "1"
    assert "External Survey" in row["prior_search_provenance"]
    assert '"tmag":12.3' in row["metrics_json"]
    assert row["expected_information_gain"] == "0.5"
    assert row["latest_pathway"] == "planet_hunters_discussion"
    assert row["meets_strict_follow_up_bar"] == "true"
    assert row["search_created_at"]
    assert row["exported_at_utc"] == exported_at.isoformat()
    assert store.get_search(search["search_id"])["targets"]


def test_integrity_reports_all_required_durable_concepts(tmp_path: Path) -> None:
    db_path = tmp_path / "hunter.sqlite3"
    store = HunterStore(db_path)
    summary = store.validity_summary()
    assert summary["ok"] is True
    assert summary["issues"] == []
    assert summary["sqlite_integrity"] == "ok"
    assert summary["foreign_key_violation_count"] == 0
    assert summary["schema_version"] == 6
    assert summary["immutable_trigger_count"] == 14
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
        "follow_up_events",
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
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("DROP TABLE follow_up_events")
    connection.execute("ALTER TABLE follow_up_registry RENAME TO follow_up_registry_v2")
    connection.execute(
        "CREATE TABLE follow_up_registry ("
        "follow_up_id TEXT PRIMARY KEY, search_id TEXT NOT NULL, target_id TEXT NOT NULL, "
        "candidate_id TEXT NOT NULL, priority REAL NOT NULL, reason TEXT NOT NULL, "
        "evidence_json TEXT NOT NULL, prior_search_provenance_json TEXT NOT NULL, "
        "recommended_action TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, "
        "FOREIGN KEY(search_id) REFERENCES search_manifests(search_id))"
    )
    connection.execute(
        "INSERT INTO follow_up_registry SELECT follow_up_id, search_id, target_id, "
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
    assert row["parent_follow_up_id"] is None
    assert row["events"][0]["state"] == "open"
    assert row["events"][0]["detail"] == {"migration_backfill": True}


def test_schema_v3_non_executable_open_row_migrates_to_deferred(tmp_path: Path) -> None:
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
                    priority=5.0,
                    reason="insufficient observations",
                    evidence={},
                    recommended_action="wait for more observations",
                    search_eligible=False,
                    revisit_reason="a new sector must cover the event",
                ),
            ),
        ),
        search_id=search["search_id"],
    )
    connection = sqlite3.connect(db_path)
    follow_up_id, source_search_id = connection.execute(
        "SELECT follow_up_id, search_id FROM follow_up_registry"
    ).fetchone()
    connection.execute("UPDATE follow_up_registry SET status='open'")
    connection.execute("DROP TRIGGER hunter_immutable_follow_up_events_delete")
    connection.execute("DELETE FROM follow_up_events")
    HunterStore._append_follow_up_event(
        connection,
        follow_up_id=str(follow_up_id),
        state="open",
        related_search_id=str(source_search_id),
        detail={"migration_backfill": True},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    connection.execute("DELETE FROM schema_migrations WHERE version=4")
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    migrated = HunterStore(db_path)
    row = migrated.list_follow_ups(status="deferred")[0]
    assert row["search_eligible"] is False
    assert [event["state"] for event in row["events"]] == ["open", "deferred"]
    assert row["events"][-1]["detail"]["migration_correction"].startswith(
        "non-executable follow-up"
    )


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
