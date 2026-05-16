"""Log-score models for each competing hypothesis.

Each function takes a CandidateFeatures and returns an unnormalised log score.
Softmax over the six scores gives the posterior probabilities (see scoring.py).

Weights and priors are from SCORING_MODEL.md §7–8 and are intentionally
conservative starting points to be calibrated later.
"""
from __future__ import annotations

import math

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Default priors (log space)
# Intentionally pessimistic about new planet candidates (§8).
# ---------------------------------------------------------------------------

DEFAULT_LOG_PRIORS: dict[str, float] = {
    "planet_candidate": math.log(0.10),
    "eclipsing_binary": math.log(0.20),
    "background_eclipsing_binary": math.log(0.20),
    "stellar_variability": math.log(0.20),
    "instrumental_artifact": math.log(0.20),
    "known_object": math.log(0.10),
}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _w(score: float | None, weight: float) -> float:
    """Weight a feature score; return 0.0 when the score is unavailable.

    Absent diagnostics contribute nothing — neither evidence for nor against
    any hypothesis.  This keeps missing features neutral rather than biasing
    the posterior in either direction.
    """
    return weight * score if score is not None else 0.0


# ---------------------------------------------------------------------------
# Hypothesis log-score functions
# ---------------------------------------------------------------------------


def log_score_planet(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["planet_candidate"],
) -> float:
    """
    Log score for the planet candidate hypothesis.

    Positive terms: signal quality, consistency, plausible morphology.
    Negative terms: false-positive indicators.
    """
    return (
        log_prior
        + _w(features.log_snr_score, 1.20)
        + _w(features.transit_count_score, 1.00)
        + _w(features.depth_consistency_score, 0.80)
        + _w(features.duration_consistency_score, 0.70)
        + _w(features.duration_plausibility_score, 0.70)
        + _w(features.transit_shape_score, 0.60)
        - _w(features.odd_even_mismatch_score, 1.50)
        - _w(features.secondary_eclipse_score, 1.80)
        - _w(features.centroid_offset_score, 1.40)
        - _w(features.contamination_score, 1.20)
        - _w(features.systematics_overlap_score, 1.10)
        - _w(features.stellar_variability_score, 0.90)
        - _w(features.depth_scatter_chi2_score, 0.60)
        - _w(features.transit_timing_variation_score, 0.50)
    )


def log_score_eclipsing_binary(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["eclipsing_binary"],
) -> float:
    """Log score for the on-target eclipsing binary hypothesis."""
    return (
        log_prior
        + _w(features.odd_even_mismatch_score, 1.80)
        + _w(features.secondary_eclipse_score, 1.70)
        + _w(features.v_shape_score, 1.40)
        + _w(features.large_depth_score, 1.20)
        + _w(features.companion_radius_too_large_score, 1.20)
        + _w(features.duration_implausibility_score, 0.80)
    )


def log_score_background_eb(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["background_eclipsing_binary"],
) -> float:
    """Log score for the background eclipsing binary hypothesis."""
    return (
        log_prior
        + _w(features.centroid_offset_score, 1.80)
        + _w(features.contamination_score, 1.60)
        + _w(features.nearby_bright_source_score, 1.20)
        + _w(features.aperture_edge_score, 1.00)
        + _w(features.dilution_sensitivity_score, 0.80)
    )


def log_score_stellar_variability(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["stellar_variability"],
) -> float:
    """Log score for the stellar variability hypothesis."""
    return (
        log_prior
        + _w(features.variability_periodogram_score, 1.50)
        + _w(features.harmonic_score, 1.20)
        + _w(features.flare_score, 1.00)
        + _w(features.quasi_periodic_score, 1.00)
        + _w(features.non_box_shape_score, 0.80)
    )


def log_score_instrumental(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["instrumental_artifact"],
) -> float:
    """Log score for the instrumental artifact hypothesis."""
    return (
        log_prior
        + _w(features.systematics_overlap_score, 1.70)
        + _w(features.quality_flag_score, 1.30)
        + _w(features.sector_boundary_score, 1.20)
        + _w(features.background_excursion_score, 1.20)
        + _w(features.single_event_score, 1.00)
        + _w(features.nearby_targets_common_signal_score, 1.00)
        + _w(features.depth_scatter_chi2_score, 0.90)
        + _w(features.transit_timing_variation_score, 0.60)
    )


def log_score_known_object(
    features: CandidateFeatures,
    log_prior: float = DEFAULT_LOG_PRIORS["known_object"],
) -> float:
    """Log score for the known catalog object hypothesis."""
    return (
        log_prior
        + _w(features.target_id_match_score, 2.50)
        + _w(features.period_match_score, 2.00)
        + _w(features.epoch_match_score, 1.50)
        + _w(features.coordinate_match_score, 1.20)
    )


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------


def compute_log_scores(
    features: CandidateFeatures,
    log_priors: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute all six hypothesis log scores for a given feature set.

    Returns a dict whose keys match the fields of HypothesisPosterior.
    Entries in log_priors override the defaults for any supplied keys.
    """
    lp = dict(DEFAULT_LOG_PRIORS)
    if log_priors is not None:
        lp.update(log_priors)

    return {
        "planet_candidate": log_score_planet(
            features, lp["planet_candidate"]
        ),
        "eclipsing_binary": log_score_eclipsing_binary(
            features, lp["eclipsing_binary"]
        ),
        "background_eclipsing_binary": log_score_background_eb(
            features, lp["background_eclipsing_binary"]
        ),
        "stellar_variability": log_score_stellar_variability(
            features, lp["stellar_variability"]
        ),
        "instrumental_artifact": log_score_instrumental(
            features, lp["instrumental_artifact"]
        ),
        "known_object": log_score_known_object(
            features, lp["known_object"]
        ),
    }
