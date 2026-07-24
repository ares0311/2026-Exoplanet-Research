"""Tests for the three EXO-Hunter production shell entry points."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from exo_toolkit.hunter_cli import (
    _commit_run_report,
    _pipeline_runner,
    _select_live_new_candidates,
    create_new_search,
    import_follow_up,
    run_new_search,
    show_follow_ups,
)
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_SELECTOR_VERSION,
    NEW_SELECTOR_VERSION,
    OPERATOR_SELECTOR_VERSION,
    selection_contract,
)
from exo_toolkit.search_lifecycle import (
    FollowUpRecommendation,
    HunterCandidate,
    HunterStore,
    PriorSearch,
    TargetExecutionResult,
)


def _candidates(count: int) -> list[HunterCandidate]:
    return [
        HunterCandidate(
            target_id=f"TIC {index}",
            canonical_id=f"TIC {index}",
            source="offline test catalog",
            source_provenance={
                "search_category": "new",
                "selector_version": NEW_SELECTOR_VERSION,
            },
            ranking_score=float(count - index),
            selection_reason="offline deterministic rank",
            metrics={"rank": index},
        )
        for index in range(count)
    ]


def test_create_run_show_complete_offline_path(tmp_path: Path, capsys: object) -> None:
    db = tmp_path / "hunter.sqlite3"

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        return _candidates(3), {"candidate_universe_returned": 3}

    assert (
        create_new_search(
            ["--targets", "2", "--mode", "new", "--db", str(db), "--json"],
            live_selector=selector,
        )
        == 0
    )

    def runner_factory(**_: object):  # type: ignore[no-untyped-def]
        def runner(candidate: HunterCandidate) -> TargetExecutionResult:
            follow_ups = (
                FollowUpRecommendation(
                    candidate_id=f"{candidate.target_id}-signal",
                    priority=90.0,
                    reason="offline evidence gate",
                    evidence={"fpp": 0.05},
                    recommended_action="review candidate packet",
                ),
            )
            return TargetExecutionResult(
                status="candidate_found",
                result={"composite_interpretation": "candidate signal"},
                provenance={"raw_uris": ["offline:fixture"]},
                follow_ups=follow_ups,
            )

        return runner

    reports: list[object] = []
    assert (
        run_new_search(
            ["--db", str(db), "--workers", "2", "--json"],
            runner_factory=runner_factory,
            report_fn=reports.append,
        )
        == 0
    )
    assert len(reports) == 1
    assert show_follow_ups(["--db", str(db), "--json"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"status": "completed"' in output
    assert '"follow_ups"' in output
    assert "review candidate packet" in output
    completed = HunterStore(db).target_history("TIC 0")[0]
    stored = HunterStore(db).get_search(completed["search_id"])
    assert stored["selector_version"] == NEW_SELECTOR_VERSION
    assert stored["config"]["selection_contract"] == selection_contract("new")


def test_create_reports_shortfall_and_still_creates_best_available_search(
    tmp_path: Path, capsys: object
) -> None:
    """A request for more targets than are available must return the best
    available N, not fail outright (directive: return fewer than N only when
    fewer than N valid candidates actually exist)."""
    db = tmp_path / "hunter.sqlite3"

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        return _candidates(1), {}

    code = create_new_search(
        ["--targets", "2", "--mode", "new", "--db", str(db), "--json"],
        live_selector=selector,
    )
    assert code == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["requested_targets"] == 2
    assert output["selected_targets"] == 1
    assert output["shortfall"] == 1
    open_searches = HunterStore(db).open_searches()
    assert len(open_searches) == 1
    assert open_searches[0]["selected_target_count"] == 1


def test_create_fails_when_zero_candidates_are_available(tmp_path: Path) -> None:
    db = tmp_path / "hunter.sqlite3"

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        candidate = _candidates(1)[0].model_copy(
            update={"eligible": False, "eligibility_reason": "already searched"}
        )
        return [candidate], {}

    code = create_new_search(
        ["--targets", "1", "--mode", "new", "--db", str(db)],
        live_selector=selector,
    )
    assert code == 2
    assert HunterStore(db).open_searches() == []


def test_live_selector_stamps_versioned_information_gain_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanner = SimpleNamespace(
        _load_toi_tic_ids=lambda **_: set(),
        _load_ctoi_tic_ids=lambda **_: set(),
        _load_confirmed_host_tic_ids=lambda **_: set(),
        _load_asassn_variable_tic_ids=lambda *_args, **_kwargs: set(),
        select_targets=lambda *_args, **_kwargs: [
            {"tic_id": 42, "priority": 0.7, "tmag": 10.0}
        ],
        inspect_target_products=lambda *_args, **_kwargs: {
            "products": ["product-1", "product-2"],
            "total_bytes": 1_000_000_000,
            "priority": 0.8,
        },
    )
    monkeypatch.setattr("exo_toolkit.hunter_cli._load_project_skill", lambda _: scanner)

    candidates, search_log = _select_live_new_candidates(
        targets=1,
        pool_size=1,
        workers=1,
        tmag_range=(6.0, 13.5),
        store=HunterStore(tmp_path / "hunter.sqlite3"),
        progress_fn=None,
    )

    candidate = candidates[0]
    assert candidate.source_provenance["selector_version"] == NEW_SELECTOR_VERSION
    assert candidate.metrics["expected_information_gain"] == pytest.approx(0.32)
    assert candidate.metrics["scientific_suitability"] == pytest.approx(0.8)
    assert candidate.ranking_score == pytest.approx(67.2)
    assert search_log["stage_two_eligible_count"] == 1
    assert search_log["discovery_expansion_attempts"] == [
        {"attempt": 0, "tmag_range": [6.0, 13.5], "raw_candidates_returned": 1}
    ]
    assert search_log["final_tmag_range"] == [6.0, 13.5]


def test_live_selector_widens_tmag_range_when_sweep_is_thin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directive: candidate pools are adaptive, never arbitrarily fixed — expand
    discovery before settling for fewer than requested."""
    calls: list[tuple[float, float]] = []

    def fake_select_targets(
        _n: int,
        *,
        tmag_range: tuple[float, float],
        exclude_tic_ids: set[int],
        full_sweep: bool,
        max_workers: int,
        search_log: dict[str, object],
    ) -> list[dict[str, object]]:
        calls.append(tmag_range)
        if tmag_range == (12.0, 14.5):
            return [{"tic_id": 1, "priority": 0.5, "tmag": 13.0}]
        return [
            {"tic_id": 1, "priority": 0.5, "tmag": 13.0},
            {"tic_id": 2, "priority": 0.6, "tmag": 12.0},
            {"tic_id": 3, "priority": 0.7, "tmag": 14.0},
        ]

    scanner = SimpleNamespace(
        _load_toi_tic_ids=lambda **_: set(),
        _load_ctoi_tic_ids=lambda **_: set(),
        _load_confirmed_host_tic_ids=lambda **_: set(),
        _load_asassn_variable_tic_ids=lambda *_args, **_kwargs: set(),
        select_targets=fake_select_targets,
        inspect_target_products=lambda *_args, **_kwargs: {
            "products": ["product-1"],
            "total_bytes": 1_000_000,
            "priority": 0.5,
        },
    )
    monkeypatch.setattr("exo_toolkit.hunter_cli._load_project_skill", lambda _: scanner)

    candidates, search_log = _select_live_new_candidates(
        targets=3,
        pool_size=10,
        workers=1,
        tmag_range=(12.0, 14.5),
        store=HunterStore(tmp_path / "hunter.sqlite3"),
        progress_fn=None,
    )

    assert calls == [(12.0, 14.5), (11.0, 15.5)]
    assert len(candidates) == 3
    attempts = search_log["discovery_expansion_attempts"]
    assert [a["raw_candidates_returned"] for a in attempts] == [1, 3]
    assert search_log["final_tmag_range"] == [11.0, 15.5]


def test_live_selector_expansion_never_raises_when_still_thin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery must never hard-fail; create_search() decides best-available-N."""
    scanner = SimpleNamespace(
        _load_toi_tic_ids=lambda **_: set(),
        _load_ctoi_tic_ids=lambda **_: set(),
        _load_confirmed_host_tic_ids=lambda **_: set(),
        _load_asassn_variable_tic_ids=lambda *_args, **_kwargs: set(),
        select_targets=lambda *_args, **_kwargs: [
            {"tic_id": 1, "priority": 0.5, "tmag": 13.0}
        ],
        inspect_target_products=lambda *_args, **_kwargs: {
            "products": ["product-1"],
            "total_bytes": 1_000_000,
            "priority": 0.5,
        },
    )
    monkeypatch.setattr("exo_toolkit.hunter_cli._load_project_skill", lambda _: scanner)

    candidates, search_log = _select_live_new_candidates(
        targets=100,
        pool_size=10,
        workers=1,
        tmag_range=(12.0, 14.5),
        store=HunterStore(tmp_path / "hunter.sqlite3"),
        progress_fn=None,
    )

    assert len(candidates) == 1
    assert len(search_log["discovery_expansion_attempts"]) == 4  # initial + 3 expansions
    assert search_log["final_tmag_range"] == [9.0, 17.5]


def test_invalid_target_count_fails_before_live_selection(tmp_path: Path) -> None:
    selector_called = False

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        nonlocal selector_called
        selector_called = True
        return _candidates(1), {}

    code = create_new_search(
        ["--targets", "0", "--mode", "new", "--db", str(tmp_path / "hunter.sqlite3")],
        live_selector=selector,
    )
    assert code == 2
    assert not selector_called


def test_more_than_100_targets_writes_timestamped_csv(tmp_path: Path) -> None:
    db = tmp_path / "hunter.sqlite3"
    manifests = tmp_path / "manifests"

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        return _candidates(102), {}

    code = create_new_search(
        [
            "--targets",
            "101",
            "--mode",
            "new",
            "--db",
            str(db),
            "--manifest-dir",
            str(manifests),
            "--json",
        ],
        live_selector=selector,
    )
    assert code == 0
    files = list(manifests.glob("exo-search-*.csv"))
    assert len(files) == 1
    assert len(files[0].read_text(encoding="utf-8").splitlines()) == 102


def test_candidate_json_file_supports_external_follow_up_provenance(tmp_path: Path) -> None:
    db = tmp_path / "hunter.sqlite3"
    candidate_file = tmp_path / "followup.json"
    candidate_file.write_text(
        json.dumps(
            [
                {
                    "target_id": "TIC 42",
                    "canonical_id": "TIC 42",
                    "source": "published external survey",
                    "source_provenance": {"search_category": "follow-up"},
                    "ranking_score": 80,
                    "selection_reason": "published unresolved event",
                    "metrics": {"follow_up_value": 3},
                    "prior_searches": [
                        {
                            "searched_by": "Researcher",
                            "searched_at": "2025-01-01T00:00:00Z",
                            "source_project": "External Project",
                            "method_or_data": "TESS sector 1",
                            "result": "unresolved candidate",
                            "provenance_uri": "doi:example",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert (
        create_new_search(
            [
                "--targets",
                "1",
                "--mode",
                "follow-up",
                "--db",
                str(db),
                "--candidate-file",
                str(candidate_file),
                "--json",
            ]
        )
        == 0
    )
    search = HunterStore(db).open_searches()[0]
    restored = HunterStore(db).get_search(search["search_id"])
    assert restored["selector_version"] == OPERATOR_SELECTOR_VERSION
    assert restored["config"]["selection_contract"] == selection_contract(
        "follow-up", operator_supplied=True
    )
    assert restored["targets"][0]["candidate"]["prior_searches"][0]["searched_by"] == "Researcher"


def test_default_follow_up_imports_and_ranks_durable_history(
    tmp_path: Path, capsys: object
) -> None:
    db = tmp_path / "hunter.sqlite3"
    history = tmp_path / "history.json"
    source = tmp_path / "logs" / "prior.json"
    source.parent.mkdir(parents=True)
    source.write_text("prior search evidence", encoding="utf-8")
    source_sha256 = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    history.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "search_id": "historical-new-search",
                        "mode": "new",
                        "started_at": "2025-01-01T00:00:00+00:00",
                        "completed_at": "2025-01-01T01:00:00+00:00",
                        "searched_by": "EXO-Hunter",
                        "source_project": "legacy project",
                        "method_or_data": "TESS QLP",
                        "source_path": "logs/prior.json",
                        "source_sha256": source_sha256,
                        "provenance_uri": "artifact:prior-new",
                        "entries": [
                            {
                                "target_id": "TIC 42",
                                "status": "candidate_found",
                                "searched_at": "2025-01-01T00:30:00+00:00",
                                "ranking_score": 0.9,
                                "metrics": {"best_fpp": 0.1},
                                "result": {"interpretation": "promising"},
                                "best_fpp": 0.1,
                                "best_detection_confidence": 0.8,
                                "best_pathway": "planet_hunters_discussion",
                                "error_message": None,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert create_new_search(
        [
            "--targets",
            "1",
            "--mode",
            "follow-up",
            "--db",
            str(db),
            "--history-manifest",
            str(history),
            "--json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["candidate_pool_count"] == 1
    stored = HunterStore(db)
    assert len(stored.target_history("TIC 42")) == 1
    assert stored.validity_summary()["ok"] is True
    search = stored.get_search(output["search_id"])
    assert search["selector_version"] == FOLLOW_UP_SELECTOR_VERSION
    assert search["config"]["selection_contract"] == selection_contract("follow-up")
    assert search["targets"][0]["target_id"] == "TIC 42"


def test_follow_up_table_shows_external_prior_search_provenance(
    tmp_path: Path, capsys: object
) -> None:
    db = tmp_path / "hunter.sqlite3"
    store = HunterStore(db)
    candidate = _candidates(1)[0].model_copy(
        update={
            "source_provenance": {"search_category": "follow-up"},
            "prior_searches": (
                PriorSearch(
                    searched_by="External Researcher",
                    searched_at=datetime(2025, 1, 1, tzinfo=UTC),
                    source_project="External Survey",
                    method_or_data="TESS sector 1",
                    result="unresolved event",
                    provenance_uri="doi:example",
                ),
            ),
        }
    )
    search = store.create_search(
        [candidate],
        requested_target_count=1,
        mode="follow-up",
        selector_version="test-v1",
        config={},
    )
    store.execute_search(
        lambda _: TargetExecutionResult(
            status="candidate_found",
            result={},
            provenance={},
            follow_ups=(
                FollowUpRecommendation(
                    candidate_id="signal-1",
                    priority=90,
                    reason="unresolved signal",
                    evidence={"fpp": 0.05},
                    recommended_action="obtain photometry",
                ),
            ),
        ),
        search_id=search["search_id"],
    )

    assert show_follow_ups(["--db", str(db), "--no-color"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "External Survey" in output
    assert "External Researcher" in output


def test_pyproject_registers_exact_required_shell_entry_points() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["Create-New-Search"].endswith(":create_new_search_entry")
    assert scripts["Run-New-Search"].endswith(":run_new_search_entry")
    assert scripts["Show-Follow-Ups"].endswith(":show_follow_ups_entry")
    assert scripts["Import-Follow-Up"].endswith(":import_follow_up_entry")


def test_import_reviewed_follow_up_verifies_sources_and_reports(
    tmp_path: Path, capsys: object
) -> None:
    source = tmp_path / "review.json"
    source.write_text('{"status":"reviewed"}', encoding="utf-8")
    source_sha = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_search_id": "prior-search",
                "source_attempt_id": "prior-attempt",
                "completed_at": "2026-01-01T00:00:00+00:00",
                "source_files": [{"path": str(source), "sha256": source_sha}],
                "candidate": {
                    **_candidates(1)[0].model_dump(mode="json"),
                    "source_provenance": {"search_category": "follow-up"},
                    "prior_searches": [
                        {
                            "searched_by": "Researcher",
                            "searched_at": "2025-01-01T00:00:00+00:00",
                            "source_project": "Survey",
                            "method_or_data": "TESS QLP",
                            "result": "plausible_but_weak",
                            "provenance_uri": str(source),
                        }
                    ],
                },
                "recommendation": {
                    "candidate_id": "signal-1",
                    "priority": 90,
                    "reason": "reviewed evidence",
                    "evidence": {"events": 2},
                    "recommended_action": "await another event",
                    "search_eligible": False,
                    "revisit_reason": "new observations required",
                },
                "source_result": {"review_status": "plausible_but_weak"},
                "source_provenance": {"review": str(source)},
            }
        ),
        encoding="utf-8",
    )
    reports: list[object] = []

    assert import_follow_up(
        ["--evidence-file", str(evidence), "--db", str(tmp_path / "hunter.db"), "--json"],
        report_fn=lambda *args: reports.append(args),
    ) == 0
    assert len(reports) == 1
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"created": true' in output
    assert show_follow_ups(
        ["--db", str(tmp_path / "hunter.db"), "--status", "deferred", "--json"]
    ) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"search_eligible": false' in output
    assert "new observations required" in output


def test_import_reviewed_follow_up_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "review.json"
    source.write_text("changed", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_search_id": "prior-search",
                "source_attempt_id": "prior-attempt",
                "completed_at": "2026-01-01T00:00:00+00:00",
                "source_files": [{"path": str(source), "sha256": "0" * 64}],
                "candidate": {},
                "recommendation": {},
                "source_result": {},
                "source_provenance": {},
            }
        ),
        encoding="utf-8",
    )
    assert import_follow_up(
        ["--evidence-file", str(evidence), "--db", str(tmp_path / "hunter.db")],
        report_fn=lambda *_: None,
    ) == 2


def test_hunter_run_report_commit_failure_is_loud(
    tmp_path: Path, capsys: object
) -> None:
    class FailedReportModule:
        @staticmethod
        def run_and_commit_report(*_: object) -> bool:
            return False

    path = tmp_path / "hunter.jsonl"
    assert _commit_run_report(FailedReportModule, object(), path) is False
    error = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "WARNING: Run Report data was preserved" in error
    assert str(path) in error


def test_installed_process_can_load_required_project_skills_outside_repo_cwd(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from exo_toolkit.hunter_cli import _load_project_skill; "
                "print(_load_project_skill('star_scanner').__name__); "
                "print(_load_project_skill('run_report').__name__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["Skills.star_scanner", "Skills.run_report"]


def test_pipeline_runner_records_no_data_as_terminal_not_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_data(*_: object, **__: object) -> list[dict[str, object]]:
        raise ValueError("No TESS light curves found for 'TIC 1'")

    monkeypatch.setattr("exo_toolkit.hunter_cli.run_pipeline", no_data)
    runner = _pipeline_runner(
        scorer="bayesian",
        model_path=None,
        cnn_checkpoint=None,
        pipeline="QLP",
        exptime="long",
    )
    outcome = runner(_candidates(1)[0])
    assert outcome.status == "no_data"
    assert outcome.result["no_data_reason"].startswith("No TESS light curves")


def test_pipeline_runner_consumes_production_nested_score_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "candidate_id": "higher-fpp",
            "scores": {
                "false_positive_probability": 0.4,
                "detection_confidence": 0.8,
            },
            "pathway": "planet_hunters_discussion",
        },
        {
            "candidate_id": "follow-up-candidate",
            "scores": {
                "false_positive_probability": 0.05,
                "detection_confidence": 0.9,
            },
            "pathway": "planet_hunters_discussion",
        },
    ]
    monkeypatch.setattr("exo_toolkit.hunter_cli.run_pipeline", lambda *_, **__: rows)
    runner = _pipeline_runner(
        scorer="bayesian",
        model_path=None,
        cnn_checkpoint=None,
        pipeline="QLP",
        exptime="long",
    )

    outcome = runner(_candidates(1)[0])

    assert outcome.result["composite_result"]["candidate_id"] == "follow-up-candidate"
    # Every detected signal gets a recommendation now (rank, not a hard quality
    # gate, decides selection); the strict production bar is reported per-row.
    by_id = {row.candidate_id: row for row in outcome.follow_ups}
    assert set(by_id) == {"higher-fpp", "follow-up-candidate"}
    assert by_id["higher-fpp"].evidence["meets_strict_production_bar"] is False
    assert by_id["follow-up-candidate"].evidence["meets_strict_production_bar"] is True
    assert by_id["follow-up-candidate"].priority > by_id["higher-fpp"].priority


def test_pipeline_runner_fails_loudly_when_required_scores_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo_toolkit.hunter_cli.run_pipeline",
        lambda *_, **__: [{"candidate_id": "invalid", "pathway": "tfop_ready"}],
    )
    runner = _pipeline_runner(
        scorer="bayesian",
        model_path=None,
        cnn_checkpoint=None,
        pipeline="QLP",
        exptime="long",
    )

    with pytest.raises(RuntimeError, match="missing required score"):
        runner(_candidates(1)[0])


def test_invalid_scorer_fails_before_starting_attempt(tmp_path: Path) -> None:
    db = tmp_path / "hunter.sqlite3"
    store = HunterStore(db)
    store.create_search(
        _candidates(1),
        requested_target_count=1,
        mode="new",
        selector_version="test-v1",
        config={},
    )
    assert run_new_search(["--db", str(db), "--scorer", "opaque-ai"]) == 2
    assert store.open_searches()[0]["state"] == "pending"


def test_historical_acceptance_chain_is_preserved_without_self_attestation() -> None:
    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v2.json").read_text(
            encoding="utf-8"
        )
    )
    reassessment = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v2_reassessment.json").read_text(
            encoding="utf-8"
        )
    )
    current = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v3.json").read_text(
            encoding="utf-8"
        )
    )
    replacement = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["core_ai_dependency"] is False
    assert acceptance["live_search_execution"]["candidate_pool_count"] >= 10_000
    assert acceptance["live_search_execution"]["live_items_failed"] == 0
    assert acceptance["reviewed_follow_up_import"]["initial_import_created"] is True
    assert acceptance["reviewed_follow_up_import"]["search_eligible"] is False
    assert reassessment["original_acceptance_id"] == acceptance["acceptance_id"]
    assert reassessment["corrected_status"] == "partial"
    assert current["status"] == "partial"
    assert current["production_database"]["schema_version"] == 4
    assert current["real_follow_up_state"]["status"] == "deferred"
    assert current["real_follow_up_state"]["event_states"] == ["open", "deferred"]
    assert current["current_archive_check"]["products_found"] == 2
    assert current["production_requirements"][
        "durable_follow_up_history_universe"
    ].startswith("gap:")
    assert current["highest_priority_gap"]["evidence"].startswith(
        "The default CLI calls follow_up_candidates()"
    )
    assert replacement["supersedes"]["partial_artifact"].endswith(
        "hunter_live_acceptance_v3.json"
    )
    assert replacement["history_contract"]["event_count"] == 608
    assert replacement["follow_up_universe"]["evaluated_target_count"] == 202
    assert replacement["follow_up_universe"]["eligible_target_count"] == 0

    report_lines = Path(
        acceptance["durable_state"]["run_report_path"]
    ).read_text(encoding="utf-8")
    assert acceptance["reviewed_follow_up_import"]["imported_search_id"] in report_lines


def test_committed_hunter_history_manifest_preserves_available_project_universe() -> None:
    from exo_toolkit.hunter_history import load_verified_history_manifest

    payload = json.loads(
        Path("data_selection/hunter_prior_search_history_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert load_verified_history_manifest(
        Path("data_selection/hunter_prior_search_history_v1.json")
    ) == payload
    sources = payload["sources"]
    events = [entry for source in sources for entry in source["entries"]]
    assert payload["schema_version"] == 1
    assert len(sources) == 7
    assert len(events) == 608
    assert len({entry["target_id"] for entry in events}) == 200
    assert {source["mode"] for source in sources} == {"new", "follow-up"}
    assert all(
        __import__("hashlib").sha256(Path(source["source_path"]).read_bytes()).hexdigest()
        == source["source_sha256"]
        for source in sources
    )
