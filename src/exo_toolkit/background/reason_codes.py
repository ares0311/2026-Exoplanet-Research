"""Stable reason codes for background automation records."""

from __future__ import annotations

from enum import StrEnum


class ReasonCode(StrEnum):
    BLOCKING_ISSUE_PENALTY_HIGH = "blocking_issue_penalty_high"
    BLOCKING_ISSUE_REQUIRES_REVIEW = "blocking_issue_requires_review"
    CALIBRATION_UNCERTAINTY = "calibration_uncertainty"
    DRY_RUN_NO_WRITE = "dry_run_no_write"
    HIGH_CONTAMINATION_RISK = "high_contamination_risk"
    KNOWN_OBJECT_BENCHMARK = "known_object_benchmark"
    KNOWN_OBJECT_REQUIRES_ANNOTATION = "known_object_requires_annotation_not_submission"
    KNOWN_TESS_EXAMPLE = "known_tess_example"
    LOW_FALSE_POSITIVE_RISK = "low_false_positive_risk_benchmark"
    MISSING_PROVENANCE = "missing_provenance"
    NEVER_REVIEWED = "never_reviewed"
    NO_FOLLOW_UP_TRIGGERS = "no_follow_up_triggers"
    NO_TARGETS_AVAILABLE = "no_targets_available"
    PREVIOUSLY_REVIEWED = "previously_reviewed"
    PRIORITY_SCORE_ABOVE_FOLLOW_UP_THRESHOLD = "priority_score_above_follow_up_threshold"
    RUN_LOCK_UNAVAILABLE = "run_lock_unavailable"
    SELECTED_FOR_RUN = "selected_for_run"
    SKIPPED_LOWER_PRIORITY = "skipped_lower_priority"
    TARGET_ID_NOT_FOUND = "target_id_not_found"
    WEAK_SIGNAL_LOW_PRIORITY = "weak_signal_low_priority"


STABLE_REASON_CODES = frozenset(code.value for code in ReasonCode)
