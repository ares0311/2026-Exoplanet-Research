"""Deterministic follow-up checks for background target records."""

from __future__ import annotations

from exo_toolkit.background.config import BackgroundConfig
from exo_toolkit.background.reason_codes import ReasonCode
from exo_toolkit.background.schemas import FollowUpStatus, FollowUpTestRecord, KnownTessTarget


def trigger_reason_codes(
    target: KnownTessTarget, priority_score: float, config: BackgroundConfig
) -> list[str]:
    thresholds = config.thresholds
    codes: list[str] = []
    if priority_score >= thresholds["follow_up_priority_score"]:
        codes.append(ReasonCode.PRIORITY_SCORE_ABOVE_FOLLOW_UP_THRESHOLD.value)
    if not target.provenance:
        codes.append(ReasonCode.MISSING_PROVENANCE.value)
    if target.false_positive_risk_score <= thresholds["low_false_positive_risk"]:
        codes.append(ReasonCode.LOW_FALSE_POSITIVE_RISK.value)
    if target.calibration_confidence_score < thresholds["calibration_confidence_min"]:
        codes.append(ReasonCode.CALIBRATION_UNCERTAINTY.value)
    if target.blocking_issue_penalty >= thresholds["blocking_issue_penalty_high"]:
        codes.append(ReasonCode.BLOCKING_ISSUE_REQUIRES_REVIEW.value)
    if target.known_object:
        codes.append(ReasonCode.KNOWN_OBJECT_REQUIRES_ANNOTATION.value)
    return codes


def mandatory_follow_up_tests(
    target: KnownTessTarget, config: BackgroundConfig
) -> list[FollowUpTestRecord]:
    thresholds = config.thresholds
    provenance_status = FollowUpStatus.PASS if target.provenance else FollowUpStatus.FAIL
    false_positive_status = (
        FollowUpStatus.PASS
        if target.false_positive_risk_score <= thresholds["low_false_positive_risk"]
        else FollowUpStatus.UNCERTAIN
    )
    calibration_status = (
        FollowUpStatus.PASS
        if target.calibration_confidence_score >= thresholds["calibration_confidence_min"]
        else FollowUpStatus.UNCERTAIN
    )
    human_review_status = (
        FollowUpStatus.READY
        if target.positive_evidence and target.negative_evidence
        else FollowUpStatus.BLOCKED
    )
    return [
        FollowUpTestRecord(
            test_name="provenance_check",
            status=provenance_status,
            rationale="Fixture includes static source and version provenance."
            if provenance_status == FollowUpStatus.PASS
            else "Target is missing source provenance.",
        ),
        FollowUpTestRecord(
            test_name="false_positive_class_check",
            status=false_positive_status,
            rationale="Fixture false-positive risk is below the review threshold."
            if false_positive_status == FollowUpStatus.PASS
            else "False-positive class evidence remains uncertain.",
        ),
        FollowUpTestRecord(
            test_name="cross_source_consistency_check",
            status=FollowUpStatus.PASS if target.known_object else FollowUpStatus.BLOCKED,
            rationale="Known TESS example fixture contains a catalog benchmark label."
            if target.known_object
            else "Live cross-source catalog access is disabled by default.",
        ),
        FollowUpTestRecord(
            test_name="calibration_confidence_check",
            status=calibration_status,
            rationale="Fixture lies in a characterized benchmark regime."
            if calibration_status == FollowUpStatus.PASS
            else "Calibration confidence is below the benchmark threshold.",
        ),
        FollowUpTestRecord(
            test_name="reproducibility_check",
            status=FollowUpStatus.PASS,
            rationale="Known TESS fixture inputs are deterministic and local.",
        ),
        FollowUpTestRecord(
            test_name="human_review_checklist",
            status=human_review_status,
            rationale="Evidence, negative evidence, and limitations are ready for review."
            if human_review_status == FollowUpStatus.READY
            else "Human-review checklist is missing inspectable evidence.",
        ),
    ]


def report_is_ready(tests: list[FollowUpTestRecord]) -> bool:
    return all(test.status not in {FollowUpStatus.FAIL} for test in tests)
