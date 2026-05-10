"""Typed records for conservative background search automation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "background_search_v0.2"


class Outcome(StrEnum):
    REVIEWED = "reviewed"
    NEEDS_FOLLOW_UP = "needs_follow_up"
    BLOCKED = "blocked"


class FollowUpStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"
    READY = "ready"


@dataclass(frozen=True)
class KnownTessTarget:
    target_id: str
    target_name: str
    mission: str
    fixture_version: str
    period_days: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float
    transit_count: int
    snr: float
    scientific_interest_score: float
    data_completeness_score: float
    false_positive_risk_score: float
    follow_up_feasibility_score: float
    calibration_confidence_score: float
    blocking_issue_penalty: float
    known_object: bool
    provenance: list[str]
    positive_evidence: list[str]
    negative_evidence: list[str]
    fixture_labels: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorityFactors:
    scientific_interest_score: float
    prior_review_penalty: float
    never_reviewed_boost: float
    data_completeness_score: float
    false_positive_risk_component: float
    follow_up_feasibility_score: float
    calibration_confidence_score: float
    blocking_issue_component: float

    def to_jsonable(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TargetPriorityEvaluation:
    target_id: str
    target_name: str
    factors: PriorityFactors
    final_score: float
    reason_codes: list[str]
    selected: bool = False
    skipped_reason_codes: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["factors"] = self.factors.to_jsonable()
        return payload


@dataclass(frozen=True)
class PrioritySummary:
    selected_target_id: str | None
    evaluations: list[TargetPriorityEvaluation]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "selected_target_id": self.selected_target_id,
            "evaluations": [evaluation.to_jsonable() for evaluation in self.evaluations],
        }


@dataclass(frozen=True)
class FollowUpTestRecord:
    test_name: str
    status: FollowUpStatus
    rationale: str

    def to_jsonable(self) -> dict[str, str]:
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SubmissionRecommendation:
    destination: str
    rank: int
    suitability_rationale: str
    risks: list[str]
    prerequisites: list[str]
    recommended_action: str
    human_approval_required: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DraftReport:
    target_id: str
    ready: bool
    sections: dict[str, str]
    blocking_issues: list[str] = field(default_factory=list)
    export_paths: dict[str, str] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HumanApprovalRecord:
    target_id: str
    approved: bool
    approver: str
    approval_scope: str
    rationale: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackgroundRunResult:
    run_id: str
    target_id: str | None
    outcome: Outcome
    ledger_written: bool
    outcome_written: bool
    priority_summary: PrioritySummary
    follow_up_tests: list[FollowUpTestRecord]
    draft_report: DraftReport | None
    submission_recommendations: list[SubmissionRecommendation]
    reason_codes: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_id": self.target_id,
            "outcome": self.outcome.value,
            "ledger_written": self.ledger_written,
            "outcome_written": self.outcome_written,
            "priority_summary": self.priority_summary.to_jsonable(),
            "follow_up_tests": [record.to_jsonable() for record in self.follow_up_tests],
            "draft_report": None if self.draft_report is None else self.draft_report.to_jsonable(),
            "submission_recommendations": [
                recommendation.to_jsonable() for recommendation in self.submission_recommendations
            ],
            "reason_codes": self.reason_codes,
        }
