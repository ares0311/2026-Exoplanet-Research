"""One-shot background search runner."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from exo_toolkit.background.config import DEFAULT_CONFIG_PATH, load_background_config
from exo_toolkit.background.fixtures import load_known_tess_examples
from exo_toolkit.background.followup import mandatory_follow_up_tests, trigger_reason_codes
from exo_toolkit.background.priority import build_priority_summary
from exo_toolkit.background.reason_codes import ReasonCode
from exo_toolkit.background.reports import (
    build_draft_report,
    build_submission_recommendations,
    export_draft_report,
)
from exo_toolkit.background.schemas import (
    BackgroundRunResult,
    HumanApprovalRecord,
    Outcome,
    PrioritySummary,
)
from exo_toolkit.background.storage import BackgroundStore


def background_run_once(
    db_path: Path,
    command: str = "exo background-run-once",
    config_path: Path = DEFAULT_CONFIG_PATH,
    target_id: str | None = None,
    dry_run: bool = False,
    export_reports: bool = True,
) -> BackgroundRunResult:
    config = load_background_config(config_path)
    store = BackgroundStore(db_path)
    all_targets = load_known_tess_examples()
    targets = (
        [target for target in all_targets if target.target_id == target_id]
        if target_id is not None
        else all_targets
    )
    run_id = f"run_{uuid4().hex}"
    started_at = _now()
    completed_at = started_at
    priority_summary = build_priority_summary(targets, store, config)
    selected_target = next(
        (target for target in targets if target.target_id == priority_summary.selected_target_id),
        None,
    )
    lock_acquired = False

    if dry_run:
        dry_run_triggers: list[str] = []
        dry_run_outcome = Outcome.BLOCKED
        if selected_target is not None:
            selected_evaluation = next(
                evaluation
                for evaluation in priority_summary.evaluations
                if evaluation.target_id == selected_target.target_id
            )
            dry_run_triggers = trigger_reason_codes(
                selected_target, selected_evaluation.final_score, config
            )
            dry_run_outcome = Outcome.NEEDS_FOLLOW_UP if dry_run_triggers else Outcome.REVIEWED
        return BackgroundRunResult(
            run_id=run_id,
            target_id=selected_target.target_id if selected_target else target_id,
            outcome=dry_run_outcome,
            ledger_written=False,
            outcome_written=False,
            priority_summary=priority_summary,
            follow_up_tests=[],
            draft_report=None,
            submission_recommendations=[],
            reason_codes=[ReasonCode.DRY_RUN_NO_WRITE.value, *dry_run_triggers],
        )

    try:
        lock_acquired = store.acquire_run_lock(
            lock_name="background_search",
            owner=run_id,
            wait_seconds=config.scheduler["lock_wait_seconds"],
            poll_seconds=config.scheduler["lock_poll_seconds"],
            acquired_at=started_at,
        )
        if not lock_acquired:
            return _write_blocked_run(
                store=store,
                run_id=run_id,
                started_at=started_at,
                completed_at=_now(),
                command=command,
                target_id="RUN_LOCK",
                config=config.ledger_config(),
                provenance=[],
                error_message="Another background run still owns the run lock.",
                reason_code=ReasonCode.RUN_LOCK_UNAVAILABLE.value,
                priority_summary=priority_summary,
            )

        if selected_target is None:
            reason_code = (
                ReasonCode.TARGET_ID_NOT_FOUND.value
                if target_id is not None
                else ReasonCode.NO_TARGETS_AVAILABLE.value
            )
            return _write_blocked_run(
                store=store,
                run_id=run_id,
                started_at=started_at,
                completed_at=_now(),
                command=command,
                target_id=target_id or "NO_TARGET",
                config=config.ledger_config(),
                provenance=[],
                error_message="No matching fixture targets are available.",
                reason_code=reason_code,
                priority_summary=priority_summary,
            )

        selected_evaluation = next(
            evaluation
            for evaluation in priority_summary.evaluations
            if evaluation.target_id == selected_target.target_id
        )
        triggers = trigger_reason_codes(selected_target, selected_evaluation.final_score, config)
        outcome = Outcome.NEEDS_FOLLOW_UP if triggers else Outcome.REVIEWED
        follow_up_tests = mandatory_follow_up_tests(selected_target, config) if triggers else []
        draft_report = build_draft_report(selected_target, follow_up_tests) if triggers else None
        if draft_report is not None and export_reports:
            draft_report = export_draft_report(
                draft_report,
                run_id=run_id,
                export_dir=Path(config.reports["export_dir"]),
                formats=list(config.reports["formats"]),
            )
        recommendations = build_submission_recommendations(selected_target) if triggers else []
        completed_at = _now()

        store.write_run_ledger(
            {
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "command": command,
                "target_id": selected_target.target_id,
                "outcome": outcome.value,
                "status": "completed",
                "error_message": None,
                "config": config.ledger_config(),
                "provenance": selected_target.provenance,
            }
        )
        store.write_priority_evaluations(
            run_id,
            [evaluation.to_jsonable() for evaluation in priority_summary.evaluations],
            completed_at,
        )

        if outcome == Outcome.NEEDS_FOLLOW_UP:
            test_payload = [test.to_jsonable() for test in follow_up_tests]
            store.write_follow_up_tests(
                run_id, selected_target.target_id, test_payload, completed_at
            )
            store.write_needs_follow_up_outcome(
                {
                    "run_id": run_id,
                    "target_id": selected_target.target_id,
                    "trigger_codes": triggers,
                    "mandatory_tests": test_payload,
                    "summary": (
                        "Fixture target requires follow-up record and human-readable review."
                    ),
                    "created_at": completed_at,
                }
            )
            if draft_report is not None:
                store.write_draft_report(run_id, draft_report.to_jsonable(), completed_at)
                store.write_report_exports(
                    run_id, selected_target.target_id, draft_report.export_paths, completed_at
                )
            store.write_submission_recommendations(
                run_id,
                selected_target.target_id,
                [recommendation.to_jsonable() for recommendation in recommendations],
                completed_at,
            )
            store.write_approval_record(
                run_id,
                HumanApprovalRecord(
                    target_id=selected_target.target_id,
                    approved=False,
                    approver="system",
                    approval_scope="external_submission",
                    rationale=(
                        "External submission remains blocked pending explicit human approval."
                    ),
                ).to_jsonable(),
                completed_at,
            )
        else:
            store.write_reviewed_outcome(
                {
                    "run_id": run_id,
                    "target_id": selected_target.target_id,
                    "reason_codes": [ReasonCode.NO_FOLLOW_UP_TRIGGERS.value],
                    "negative_evidence": selected_target.negative_evidence,
                    "summary": (
                        "Fixture target was reviewed with no follow-up triggers. Negative evidence "
                        "and low-priority factors are preserved for audit."
                    ),
                    "created_at": completed_at,
                }
            )

        return BackgroundRunResult(
            run_id=run_id,
            target_id=selected_target.target_id,
            outcome=outcome,
            ledger_written=True,
            outcome_written=True,
            priority_summary=priority_summary,
            follow_up_tests=follow_up_tests,
            draft_report=draft_report,
            submission_recommendations=recommendations,
            reason_codes=triggers if triggers else [ReasonCode.NO_FOLLOW_UP_TRIGGERS.value],
        )
    finally:
        if lock_acquired:
            store.release_run_lock("background_search", run_id)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_blocked_run(
    store: BackgroundStore,
    run_id: str,
    started_at: str,
    completed_at: str,
    command: str,
    target_id: str,
    config: dict[str, str | bool],
    provenance: list[str],
    error_message: str,
    reason_code: str,
    priority_summary: PrioritySummary,
) -> BackgroundRunResult:
    store.write_run_ledger(
        {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "command": command,
            "target_id": target_id,
            "outcome": Outcome.BLOCKED.value,
            "status": "blocked",
            "error_message": error_message,
            "config": config,
            "provenance": provenance,
        }
    )
    store.write_reviewed_outcome(
        {
            "run_id": run_id,
            "target_id": target_id,
            "reason_codes": [reason_code],
            "negative_evidence": [error_message],
            "summary": "Run was blocked before target follow-up; blocker is preserved for audit.",
            "created_at": completed_at,
        }
    )
    return BackgroundRunResult(
        run_id=run_id,
        target_id=target_id,
        outcome=Outcome.BLOCKED,
        ledger_written=True,
        outcome_written=True,
        priority_summary=priority_summary,
        follow_up_tests=[],
        draft_report=None,
        submission_recommendations=[],
        reason_codes=[reason_code],
    )
