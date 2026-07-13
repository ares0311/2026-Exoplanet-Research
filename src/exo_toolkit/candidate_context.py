"""Empirical candidate-score context from a labeled calibration reference."""
from __future__ import annotations

import bisect
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateContextReference(BaseModel):
    """Frozen empirical reference distribution for one production score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    context_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    calibration_dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    score_name: Literal["full_ensemble_planet_probability"]
    threshold_version: Literal["no_decision_threshold_v1"]
    decision_threshold: None = None
    scores_ascending: tuple[float, ...]
    labels_aligned: tuple[Literal[0, 1], ...]
    n_samples: int = Field(gt=0)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    source_predictions_path: str = Field(min_length=1)
    source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stacking_weights_path: str = Field(min_length=1)
    stacking_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_reference(self) -> CandidateContextReference:
        """Require aligned, sorted, internally consistent reference samples."""
        if len(self.scores_ascending) != self.n_samples:
            raise ValueError("scores_ascending length must equal n_samples")
        if len(self.labels_aligned) != self.n_samples:
            raise ValueError("labels_aligned length must equal n_samples")
        if any(not 0.0 <= score <= 1.0 for score in self.scores_ascending):
            raise ValueError("reference scores must be in [0, 1]")
        if any(
            left > right
            for left, right in zip(
                self.scores_ascending, self.scores_ascending[1:], strict=False
            )
        ):
            raise ValueError("scores_ascending must be sorted")
        if self.n_positive != sum(self.labels_aligned):
            raise ValueError("n_positive does not match labels")
        if self.n_negative != self.n_samples - self.n_positive:
            raise ValueError("n_negative does not match labels")
        return self


def load_candidate_context(path: Path | str) -> CandidateContextReference:
    """Load and validate a committed candidate-context reference."""
    return CandidateContextReference.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def contextualize_score(
    raw_score: float, reference: CandidateContextReference
) -> dict[str, object]:
    """Return empirical rank/FDR context without inventing calibration or thresholds."""
    if not 0.0 <= raw_score <= 1.0:
        raise ValueError("raw_score must be in [0, 1]")
    quantile_index = bisect.bisect_right(reference.scores_ascending, raw_score)
    tail_index = bisect.bisect_left(reference.scores_ascending, raw_score)
    tail_labels = reference.labels_aligned[tail_index:]
    tail_n = len(tail_labels)
    tail_negative = tail_n - sum(tail_labels)
    false_discovery = tail_negative / tail_n if tail_n else None
    return {
        "raw_score": raw_score,
        "calibrated_score": None,
        "score_quantile": quantile_index / reference.n_samples,
        "calibration_dataset_id": reference.calibration_dataset_id,
        "threshold_version": reference.threshold_version,
        "decision_threshold": None,
        "false_discovery_estimate": false_discovery,
        "false_discovery_reference_n": tail_n,
        "false_discovery_reference_negatives": tail_negative,
        "candidate_context_id": reference.context_id,
        "candidate_context_score_name": reference.score_name,
        "candidate_context_limitations": list(reference.limitations),
    }
