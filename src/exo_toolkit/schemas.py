"""Typed data contracts for the exoplanet candidate scoring pipeline."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Score = Annotated[float, Field(ge=0.0, le=1.0)]
OptScore = Annotated[float | None, Field(ge=0.0, le=1.0)]

Mission = Literal["TESS", "Kepler", "K2"]

SubmissionPathway = Literal[
    "known_object_annotation",
    "tfop_ready",
    "planet_hunters_discussion",
    "kepler_archive_candidate",
    "github_only_reproducibility",
    "paper_or_preprint_candidate",
]


# ---------------------------------------------------------------------------
# Input signal
# ---------------------------------------------------------------------------


class CandidateSignal(BaseModel):
    """Raw transit signal parameters as output by the BLS search stage."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    mission: Mission
    target_id: str
    period_days: float = Field(gt=0.0)
    epoch_bjd: float
    duration_hours: float = Field(gt=0.0)
    depth_ppm: float = Field(gt=0.0)
    transit_count: int = Field(ge=1)
    snr: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Feature scores
# ---------------------------------------------------------------------------


class CandidateFeatures(BaseModel):
    """
    Normalized feature scores in [0, 1] extracted from the light curve and signal.
    None means the feature could not be computed due to missing data or diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    # Detection quality
    snr_score: OptScore = None
    log_snr_score: OptScore = None
    transit_count_score: OptScore = None
    depth_consistency_score: OptScore = None
    duration_consistency_score: OptScore = None
    duration_plausibility_score: OptScore = None
    transit_shape_score: OptScore = None
    data_gap_overlap_score: OptScore = None
    transit_timing_variation_score: OptScore = None
    out_of_transit_scatter_score: OptScore = None
    multi_sector_depth_consistency_score: OptScore = None
    stellar_density_consistency_score: OptScore = None
    limb_darkening_plausibility_score: OptScore = None

    # Eclipsing binary indicators
    odd_even_mismatch_score: OptScore = None
    secondary_eclipse_score: OptScore = None
    v_shape_score: OptScore = None
    large_depth_score: OptScore = None
    companion_radius_too_large_score: OptScore = None
    duration_implausibility_score: OptScore = None

    # Background eclipsing binary indicators
    centroid_offset_score: OptScore = None
    contamination_score: OptScore = None
    nearby_bright_source_score: OptScore = None
    aperture_edge_score: OptScore = None
    dilution_sensitivity_score: OptScore = None
    centroid_motion_score: OptScore = None

    # Stellar variability indicators
    stellar_variability_score: OptScore = None
    variability_periodogram_score: OptScore = None
    harmonic_score: OptScore = None
    flare_score: OptScore = None
    quasi_periodic_score: OptScore = None
    non_box_shape_score: OptScore = None

    # Instrumental artifact indicators
    systematics_overlap_score: OptScore = None
    quality_flag_score: OptScore = None
    sector_boundary_score: OptScore = None
    background_excursion_score: OptScore = None
    single_event_score: OptScore = None
    nearby_targets_common_signal_score: OptScore = None
    depth_scatter_chi2_score: OptScore = None

    # Known-object indicators
    known_object_score: OptScore = None
    target_id_match_score: OptScore = None
    period_match_score: OptScore = None
    epoch_match_score: OptScore = None
    coordinate_match_score: OptScore = None


# ---------------------------------------------------------------------------
# Scoring outputs
# ---------------------------------------------------------------------------


class HypothesisPosterior(BaseModel):
    """
    Posterior probability for each competing hypothesis.
    All six values must sum to approximately 1.0 (tolerance ±0.01).
    """

    model_config = ConfigDict(frozen=True)

    planet_candidate: Score = 0.0
    eclipsing_binary: Score = 0.0
    background_eclipsing_binary: Score = 0.0
    stellar_variability: Score = 0.0
    instrumental_artifact: Score = 0.0
    known_object: Score = 0.0

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> HypothesisPosterior:
        total = (
            self.planet_candidate
            + self.eclipsing_binary
            + self.background_eclipsing_binary
            + self.stellar_variability
            + self.instrumental_artifact
            + self.known_object
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Posterior probabilities must sum to 1.0, got {total:.4f}"
            )
        return self


class CandidateScores(BaseModel):
    """Derived scores summarizing detection quality and scientific interest."""

    model_config = ConfigDict(frozen=True)

    false_positive_probability: Score = 0.0
    detection_confidence: Score = 0.0
    novelty_score: Score = 0.0
    habitability_interest: Score = 0.0
    followup_value: Score = 0.0
    submission_readiness: Score = 0.0


# ---------------------------------------------------------------------------
# Explanation and metadata
# ---------------------------------------------------------------------------


class CandidateExplanation(BaseModel):
    """Human-readable evidence for and against this candidate."""

    model_config = ConfigDict(frozen=True)

    positive_evidence: tuple[str, ...] = ()
    negative_evidence: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()


class ScoringMetadata(BaseModel):
    """Provenance and version info for a scoring run, required for reproducibility."""

    model_config = ConfigDict(frozen=True)

    scoring_model_name: str
    scoring_model_version: str
    scoring_model_commit: str
    threshold_config_hash: str


# ---------------------------------------------------------------------------
# Full pipeline output
# ---------------------------------------------------------------------------


class ScoredCandidate(BaseModel):
    """Canonical pipeline output for a single transit signal candidate."""

    model_config = ConfigDict(frozen=True)

    signal: CandidateSignal
    features: CandidateFeatures
    posterior: HypothesisPosterior
    scores: CandidateScores
    recommended_pathway: SubmissionPathway
    secondary_pathway: SubmissionPathway | None = None
    explanation: CandidateExplanation
    metadata: ScoringMetadata
