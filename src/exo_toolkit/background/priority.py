"""Target priority scoring for background search runs."""

from __future__ import annotations

from typing import Protocol

from exo_toolkit.background.config import BackgroundConfig
from exo_toolkit.background.reason_codes import ReasonCode
from exo_toolkit.background.schemas import (
    KnownTessTarget,
    PriorityFactors,
    PrioritySummary,
    TargetPriorityEvaluation,
)


class TargetHistoryStore(Protocol):
    def has_seen_target(self, target_id: str) -> bool: ...


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def score_target(
    target: KnownTessTarget, previously_reviewed: bool, config: BackgroundConfig
) -> TargetPriorityEvaluation:
    weights = config.priority_weights
    thresholds = config.thresholds
    prior_review_penalty = weights["prior_review_penalty"] if previously_reviewed else 0.0
    never_reviewed_boost = weights["never_reviewed_boost"] if not previously_reviewed else 0.0
    factors = PriorityFactors(
        scientific_interest_score=_clip(target.scientific_interest_score),
        prior_review_penalty=prior_review_penalty,
        never_reviewed_boost=never_reviewed_boost,
        data_completeness_score=_clip(target.data_completeness_score),
        false_positive_risk_component=_clip(1.0 - target.false_positive_risk_score),
        follow_up_feasibility_score=_clip(target.follow_up_feasibility_score),
        calibration_confidence_score=_clip(target.calibration_confidence_score),
        blocking_issue_component=_clip(1.0 - target.blocking_issue_penalty),
    )
    final_score = _clip(
        weights["scientific_interest"] * factors.scientific_interest_score
        - factors.prior_review_penalty
        + factors.never_reviewed_boost
        + weights["data_completeness"] * factors.data_completeness_score
        + weights["false_positive_risk_inverse"] * factors.false_positive_risk_component
        + weights["follow_up_feasibility"] * factors.follow_up_feasibility_score
        + weights["calibration_confidence"] * factors.calibration_confidence_score
        + weights["blocking_issue_inverse"] * factors.blocking_issue_component
    )
    reason_codes = [ReasonCode.KNOWN_TESS_EXAMPLE.value]
    if target.known_object:
        reason_codes.append(ReasonCode.KNOWN_OBJECT_BENCHMARK.value)
    if previously_reviewed:
        reason_codes.append(ReasonCode.PREVIOUSLY_REVIEWED.value)
    else:
        reason_codes.append(ReasonCode.NEVER_REVIEWED.value)
    if target.blocking_issue_penalty > thresholds["blocking_issue_penalty_high"]:
        reason_codes.append(ReasonCode.BLOCKING_ISSUE_PENALTY_HIGH.value)
    if target.false_positive_risk_score > 0.7:
        reason_codes.append(ReasonCode.HIGH_CONTAMINATION_RISK.value)
    if target.snr < 5:
        reason_codes.append(ReasonCode.WEAK_SIGNAL_LOW_PRIORITY.value)
    return TargetPriorityEvaluation(
        target_id=target.target_id,
        target_name=target.target_name,
        factors=factors,
        final_score=round(final_score, 6),
        reason_codes=reason_codes,
    )


def build_priority_summary(
    targets: list[KnownTessTarget], store: TargetHistoryStore, config: BackgroundConfig
) -> PrioritySummary:
    evaluations = [
        score_target(
            target,
            previously_reviewed=store.has_seen_target(target.target_id),
            config=config,
        )
        for target in targets
    ]
    selected = max(evaluations, key=lambda item: item.final_score, default=None)
    if selected is None:
        return PrioritySummary(selected_target_id=None, evaluations=[])
    marked = [
        TargetPriorityEvaluation(
            target_id=evaluation.target_id,
            target_name=evaluation.target_name,
            factors=evaluation.factors,
            final_score=evaluation.final_score,
            reason_codes=evaluation.reason_codes,
            selected=evaluation.target_id == selected.target_id,
            skipped_reason_codes=[]
            if evaluation.target_id == selected.target_id
            else [ReasonCode.SKIPPED_LOWER_PRIORITY.value],
        )
        for evaluation in evaluations
    ]
    return PrioritySummary(selected_target_id=selected.target_id, evaluations=marked)
