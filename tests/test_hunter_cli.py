"""Tests for the three EXO-Hunter production shell entry points."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exo_toolkit.hunter_cli import (
    _pipeline_runner,
    create_new_search,
    run_new_search,
    show_follow_ups,
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
            source_provenance={"search_category": "new"},
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


def test_create_failure_is_nonzero_and_leaves_no_manifest(tmp_path: Path) -> None:
    db = tmp_path / "hunter.sqlite3"

    def selector(**_: object) -> tuple[list[HunterCandidate], dict[str, object]]:
        return _candidates(1), {}

    code = create_new_search(
        ["--targets", "2", "--mode", "new", "--db", str(db)],
        live_selector=selector,
    )
    assert code == 2
    assert HunterStore(db).open_searches() == []


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
    assert restored["targets"][0]["candidate"]["prior_searches"][0]["searched_by"] == "Researcher"


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
    assert [row.candidate_id for row in outcome.follow_ups] == ["follow-up-candidate"]


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


def test_live_acceptance_manifest_covers_prod_contract_and_run_reports() -> None:
    acceptance = json.loads(
        Path("artifacts/manifests/hunter_live_acceptance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance["status"] == "pass"
    assert acceptance["core_ai_dependency"] is False
    assert acceptance["new_search"]["candidate_pool_count"] >= 10_000
    assert acceptance["new_search"]["execution_status"] == "completed"
    corrected = acceptance["corrective_follow_up_search"]
    assert corrected["execution_status"] == "completed"
    assert corrected["items_failed"] == 0
    assert corrected["composite_candidate"] == "TIC_237884073_s02"
    assert corrected["prior_search_id"] == acceptance["new_search"]["search_id"]
    assert all(
        str(evidence).startswith("pass:")
        for evidence in acceptance["prod_requirements"].values()
    )

    report_lines = Path(
        acceptance["durable_state"]["run_report_path"]
    ).read_text(encoding="utf-8")
    assert acceptance["new_search"]["search_id"] in report_lines
    assert corrected["search_id"] in report_lines
