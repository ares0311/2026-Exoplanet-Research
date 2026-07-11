"""Strict provenance contract for reproducible candidate-ledger records."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CandidateLedgerRecord(BaseModel):
    """One append-only candidate result with enough context to regenerate it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    project: Literal["2026 Exoplanet Research"] = "2026 Exoplanet Research"
    source_dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    target_id: str = Field(min_length=1)
    mission: Literal["TESS", "Kepler", "K2", "JWST"]
    time_window: str = Field(min_length=1)
    raw_uri: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)
    candidate_generator: str = Field(min_length=1)
    candidate_generator_params: dict[str, Any]
    model_versions: dict[str, str]
    model_scores: dict[str, float | None]
    calibrated_scores: dict[str, float | None]
    score_quantiles: dict[str, float | None]
    injection_context: dict[str, Any]
    nearest_known_artifacts: tuple[str, ...]
    review_status: Literal[
        "unreviewed",
        "artifact",
        "likely_false_positive",
        "plausible_but_weak",
        "follow_up_worthy",
        "preprocessing_failure",
        "duplicate",
        "injected_control",
    ] = "unreviewed"
    review_notes: str = ""
    regeneration_command: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
