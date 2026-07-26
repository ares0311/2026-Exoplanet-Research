"""Typed data contracts for the durable EXO-Hunter lifecycle."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["new", "follow-up"]
TargetStatus = Literal["candidate_found", "no_signal", "no_data", "failed"]
ValidityState = Literal[
    "valid",
    "stale-but-usable",
    "refresh-required",
    "invalid",
    "unknown",
]
USABLE_VALIDITY_STATES = frozenset({"valid", "stale-but-usable"})
TERMINAL_TARGET_STATUSES = frozenset({"candidate_found", "no_signal", "no_data"})
EXECUTABLE_SEARCH_STATES = frozenset({"pending", "running", "partial", "failed"})


class DecisionValidity(BaseModel):
    """Applicability assessment for evidence used in a production decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ValidityState
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    as_of: datetime | None = None
    retrieved_at: datetime
    assessed_at: datetime
    transformations: tuple[str, ...] = ()
    basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> DecisionValidity:
        if self.retrieved_at.tzinfo is None or self.assessed_at.tzinfo is None:
            raise ValueError("validity timestamps must include a timezone")
        if self.as_of is not None and self.as_of.tzinfo is None:
            raise ValueError("validity as_of must include a timezone")
        if self.assessed_at < self.retrieved_at:
            raise ValueError("validity assessed_at cannot precede retrieved_at")
        return self


class PriorSearch(BaseModel):
    """One provenance-complete search performed by any reliable project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    searched_by: str = Field(min_length=1)
    searched_at: datetime
    source_project: str = Field(min_length=1)
    method_or_data: str = Field(min_length=1)
    result: str = Field(min_length=1)
    provenance_uri: str = Field(min_length=1)
    decision_validity: DecisionValidity | None = None


class HunterCandidate(BaseModel):
    """Normalized candidate-universe row frozen into a search snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str = Field(min_length=1)
    canonical_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    mission: Literal["TESS", "Kepler", "K2", "JWST"] = "TESS"
    object_classification: str = Field(default="star", min_length=1)
    source: str = Field(min_length=1)
    source_provenance: dict[str, Any]
    decision_validity: DecisionValidity | None = None
    eligible: bool = True
    eligibility_reason: str = Field(default="eligible", min_length=1)
    distance_pc: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    estimated_download_gb: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    ranking_score: float = Field(allow_inf_nan=False)
    selection_reason: str = Field(min_length=1)
    metrics: dict[str, float | int | str | None]
    prior_searches: tuple[PriorSearch, ...] = ()

    @model_validator(mode="after")
    def provenance_is_complete(self) -> HunterCandidate:
        if not self.source_provenance:
            raise ValueError("candidate source_provenance must not be empty")
        category = self.source_provenance.get("search_category")
        if category == "follow-up" and not self.prior_searches:
            raise ValueError("follow-up candidates require prior_searches provenance")
        return self


class ArtifactIdentity(BaseModel):
    """Content identity for one model, calibration, or scoring artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ExecutionProvenance(BaseModel):
    """Required provenance for every live target outcome, including failures."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    candidate_snapshot: HunterCandidate
    pipeline_context: dict[str, Any]
    code_version: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    scorer: str = Field(min_length=1)
    model_artifacts: tuple[ArtifactIdentity, ...] = ()
    runner: str | None = None
    failure_stage: str | None = None

    @model_validator(mode="after")
    def model_backed_scorers_have_artifacts(self) -> ExecutionProvenance:
        required_roles = {
            "xgboost": {"xgboost_model"},
            "ensemble": {"xgboost_model"},
            "cnn": {"cnn_checkpoint"},
            "full-ensemble": {"xgboost_model", "cnn_checkpoint"},
        }.get(self.scorer, set())
        actual_roles = {artifact.role for artifact in self.model_artifacts}
        missing = sorted(required_roles - actual_roles)
        if missing:
            raise ValueError(
                f"model-backed scorer {self.scorer!r} is missing artifact identities: {missing}"
            )
        return self


class FollowUpRecommendation(BaseModel):
    """Validated evidence and action for one follow-up registry entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    priority: float = Field(allow_inf_nan=False)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any]
    recommended_action: str = Field(min_length=1)
    search_eligible: bool = True
    revisit_reason: str | None = None

    @model_validator(mode="after")
    def deferred_follow_up_requires_reason(self) -> FollowUpRecommendation:
        if not self.search_eligible and not self.revisit_reason:
            raise ValueError("non-executable follow-up requires revisit_reason")
        return self


class TargetExecutionResult(BaseModel):
    """One target's complete acquisition-through-interpretation outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: TargetStatus
    result: dict[str, Any]
    provenance: ExecutionProvenance | dict[str, Any]
    error_message: str | None = None
    follow_ups: tuple[FollowUpRecommendation, ...] = ()

    @model_validator(mode="after")
    def failure_requires_error_and_stage(self) -> TargetExecutionResult:
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed target result requires error_message")
        if (
            self.status == "failed"
            and isinstance(self.provenance, ExecutionProvenance)
            and not self.provenance.failure_stage
        ):
            raise ValueError("failed target result requires provenance.failure_stage")
        return self


class SearchExecutionSummary(BaseModel):
    """Structured result returned after one resumable run attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    search_id: str
    attempt_id: str
    status: Literal["completed", "partial", "failed"]
    started_at: datetime
    completed_at: datetime
    targets_total: int
    targets_already_complete: int
    targets_processed: int
    targets_succeeded: int
    targets_failed: int
    follow_ups_registered: int
