"""Posterior calculation and derived candidate scores.

Pipeline:
    CandidateFeatures
        → compute_log_scores()      (hypotheses.py)
        → softmax                   (this module)
        → HypothesisPosterior
        → compute_scores()          (this module)
        → CandidateScores

Public entry point: score_candidate()
"""
from __future__ import annotations

import math

from exo_toolkit.hypotheses import compute_log_scores
from exo_toolkit.schemas import (
    CandidateFeatures,
    CandidateScores,
    CandidateSignal,
    HypothesisPosterior,
)

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _w(score: float | None, weight: float) -> float:
    """Weighted feature contribution; 0.0 when the score is absent."""
    return weight * score if score is not None else 0.0


# ---------------------------------------------------------------------------
# Softmax / posterior
# ---------------------------------------------------------------------------


def softmax(log_scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a dict of log scores."""
    max_val = max(log_scores.values())
    exps = {k: math.exp(v - max_val) for k, v in log_scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def compute_posterior(
    features: CandidateFeatures,
    log_priors: dict[str, float] | None = None,
) -> HypothesisPosterior:
    """Compute the normalised posterior over all six hypotheses."""
    log_scores = compute_log_scores(features, log_priors)
    posteriors = softmax(log_scores)
    return HypothesisPosterior(**posteriors)


# ---------------------------------------------------------------------------
# Derived score helpers
# ---------------------------------------------------------------------------


def _detection_confidence(features: CandidateFeatures) -> float:
    """
    Sigmoid of a weighted linear combination of detection-quality features.

    Answers: "Is there a real, repeatable transit-like signal?"
    Returns 0.5 when no features are available (maximum uncertainty).
    Absent features contribute 0 (neutral), not a penalty.
    """
    raw = (
        +_w(features.log_snr_score, 1.3)
        + _w(features.transit_count_score, 1.1)
        + _w(features.depth_consistency_score, 0.8)
        + _w(features.duration_consistency_score, 0.7)
        - _w(features.systematics_overlap_score, 1.0)
        - _w(features.data_gap_overlap_score, 0.8)
    )
    return _sigmoid(raw)


def _novelty_score(features: CandidateFeatures) -> float:
    """
    1 - known_object_score.
    Returns 1.0 when no catalog matching has been attempted (assumed novel).
    """
    if features.known_object_score is not None:
        return _clip(1.0 - features.known_object_score)
    return 1.0


def _habitability_interest(
    signal: CandidateSignal,
    features: CandidateFeatures,
    posterior: HypothesisPosterior,
) -> float:
    """
    Very rough v0 habitability-interest proxy.

    NOT calibrated.  Requires stellar parameters (Teff, R_star, luminosity)
    for meaningful values — those inputs do not yet exist in the pipeline.

    Logic:
    - Longer periods (up to ~400 days) score higher as a rough insolation proxy
      for solar-type hosts.
    - Score is gated by posterior.planet_candidate — unknown signals score 0.
    - Stellar activity and contamination reduce the score.
    """
    p = signal.period_days
    if p < 5.0:
        period_factor = 0.05
    elif p < 50.0:
        period_factor = _clip((p - 5.0) / 45.0 * 0.35)
    elif p < 300.0:
        period_factor = _clip(0.35 + (p - 50.0) / 250.0 * 0.65)
    else:
        period_factor = _clip(1.0 - (p - 300.0) / 300.0)

    planet_weight = posterior.planet_candidate
    host_penalty = _w(features.stellar_variability_score, 0.40) + _w(
        features.contamination_score, 0.30
    )
    quality = _clip(planet_weight - host_penalty)
    return _clip(period_factor * quality)


def _followup_value(
    detection_confidence: float,
    novelty_score: float,
    habitability_interest: float,
    false_positive_probability: float,
) -> float:
    """
    Estimated scientific value of follow-up observations.

    host_observability_score and ephemeris_quality_score are omitted in v0
    (those inputs are not yet available in the pipeline).
    """
    return _clip(
        0.30 * detection_confidence
        + 0.25 * novelty_score
        + 0.20 * habitability_interest
        - 0.30 * false_positive_probability
    )


def _submission_readiness(
    detection_confidence: float,
    planet_candidate_posterior: float,
    novelty_score: float,
    false_positive_probability: float,
) -> float:
    """
    Readiness for external community attention.

    provenance_score and report_completeness_score are omitted in v0.
    """
    return _clip(
        0.35 * detection_confidence
        + 0.25 * planet_candidate_posterior
        + 0.15 * novelty_score
        - 0.25 * false_positive_probability
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scores(
    signal: CandidateSignal,
    features: CandidateFeatures,
    posterior: HypothesisPosterior,
) -> CandidateScores:
    """Compute all derived scores given a posterior and feature set."""
    fpp = _clip(1.0 - posterior.planet_candidate)
    dc = _detection_confidence(features)
    ns = _novelty_score(features)
    hi = _habitability_interest(signal, features, posterior)
    fv = _followup_value(dc, ns, hi, fpp)
    sr = _submission_readiness(dc, posterior.planet_candidate, ns, fpp)

    return CandidateScores(
        false_positive_probability=fpp,
        detection_confidence=dc,
        novelty_score=ns,
        habitability_interest=hi,
        followup_value=fv,
        submission_readiness=sr,
    )


def score_candidate(
    signal: CandidateSignal,
    features: CandidateFeatures,
    log_priors: dict[str, float] | None = None,
) -> tuple[HypothesisPosterior, CandidateScores]:
    """
    Full scoring pipeline: features → posterior → derived scores.

    Args:
        signal:     Raw transit signal parameters.
        features:   Extracted feature scores (None = not available).
        log_priors: Optional per-hypothesis log-prior overrides.

    Returns:
        (HypothesisPosterior, CandidateScores)
    """
    posterior = compute_posterior(features, log_priors)
    scores = compute_scores(signal, features, posterior)
    return posterior, scores
