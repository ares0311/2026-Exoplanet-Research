"""Submission pathway classification.

Routes a scored candidate to one of the six SubmissionPathways defined in
SCORING_MODEL.md §10-11 based on posterior probabilities, derived scores,
raw signal parameters, and feature diagnostics.

Three inputs are not yet computed by the v0 pipeline:
  - provenance_score       (blocks tfop_ready when absent; default 0.0)
  - multi_planet_interest  (used for paper/preprint gate; default 0.0)
  - methodological_novelty (used for paper/preprint gate; default 0.0)

Callers with access to these values may supply them explicitly.  The
conservative defaults ensure that v0 never grants a higher-tier pathway
based on missing evidence (SCORING_MODEL.md §15 Guardrails).

Note: the paper_or_preprint_candidate branch is currently unreachable via
normal pipeline inputs because Mission is constrained to TESS / Kepler / K2,
all of which have dedicated branches that always return early.  The code is
included for spec fidelity and future missions.
"""
from __future__ import annotations

from exo_toolkit.schemas import (
    CandidateFeatures,
    CandidateScores,
    CandidateSignal,
    HypothesisPosterior,
    SubmissionPathway,
)

# ---------------------------------------------------------------------------
# Classification thresholds  (SCORING_MODEL.md §11)
# ---------------------------------------------------------------------------

_KNOWN_OBJECT_THRESHOLD: float = 0.80
_HIGH_FPP_THRESHOLD: float = 0.70
_MIN_TRANSIT_COUNT: int = 2

_TFOP_PLANET_POSTERIOR: float = 0.65
_TFOP_FPP: float = 0.35
_TFOP_SNR: float = 8.0
_TFOP_CONTAMINATION: float = 0.50
_TFOP_SECONDARY_ECLIPSE: float = 0.40
_TFOP_ODD_EVEN: float = 0.40
_TFOP_PROVENANCE: float = 0.80

_KEPLER_PLANET_POSTERIOR: float = 0.65
_KEPLER_NOVELTY: float = 0.70
_KEPLER_FPP: float = 0.35

_PAPER_PLANET_POSTERIOR: float = 0.80
_PAPER_NOVELTY: float = 0.80
_PAPER_FPP: float = 0.20
_PAPER_INTEREST: float = 0.70

_DC_PLANET_HUNTERS: float = 0.45


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_submission_pathway(
    signal: CandidateSignal,
    features: CandidateFeatures,
    posterior: HypothesisPosterior,
    scores: CandidateScores,
    *,
    provenance_score: float = 0.0,
    multi_planet_interest: float = 0.0,
    methodological_novelty: float = 0.0,
) -> SubmissionPathway:
    """Classify a scored candidate into the most appropriate submission pathway.

    Args:
        signal: Raw transit signal parameters (period, SNR, mission, …).
        features: Extracted feature scores; None values mean the diagnostic
            was not run.  None feature scores are treated conservatively —
            they are assumed to *fail* any threshold gate that requires them.
        posterior: Normalised hypothesis posterior from compute_posterior().
        scores: Derived scores from compute_scores().
        provenance_score: Completeness of the data provenance record.  Not
            yet computed by the v0 pipeline; defaults to 0.0 (blocks
            tfop_ready).
        multi_planet_interest: Evidence of additional transiting planets.
            Not yet computed in v0; defaults to 0.0.
        methodological_novelty: Scientific novelty of the detection method.
            Not yet computed in v0; defaults to 0.0.

    Returns:
        SubmissionPathway label.
    """
    fpp = scores.false_positive_probability
    dc = scores.detection_confidence
    ns = scores.novelty_score
    hi = scores.habitability_interest
    p_planet = posterior.planet_candidate
    p_known = posterior.known_object

    # Feature diagnostics used as gate conditions.
    # None → conservative interpretation: gate is NOT satisfied.
    contamination = features.contamination_score
    secondary = features.secondary_eclipse_score
    odd_even = features.odd_even_mismatch_score

    # ------------------------------------------------------------------
    # Gate 1: Strong known-object match — annotate rather than re-submit
    # ------------------------------------------------------------------
    if p_known >= _KNOWN_OBJECT_THRESHOLD:
        return "known_object_annotation"

    # ------------------------------------------------------------------
    # Gate 2: High false-positive probability — exploratory log only
    # ------------------------------------------------------------------
    if fpp >= _HIGH_FPP_THRESHOLD:
        return "github_only_reproducibility"

    # ------------------------------------------------------------------
    # Gate 3: Insufficient transit count for formal submission
    # ------------------------------------------------------------------
    if signal.transit_count < _MIN_TRANSIT_COUNT:
        return "planet_hunters_discussion"

    # ------------------------------------------------------------------
    # Mission-specific routing
    # ------------------------------------------------------------------
    if signal.mission == "TESS":
        if (
            p_planet >= _TFOP_PLANET_POSTERIOR
            and fpp <= _TFOP_FPP
            and signal.snr >= _TFOP_SNR
            and contamination is not None
            and contamination < _TFOP_CONTAMINATION
            and secondary is not None
            and secondary < _TFOP_SECONDARY_ECLIPSE
            and odd_even is not None
            and odd_even < _TFOP_ODD_EVEN
            and provenance_score >= _TFOP_PROVENANCE
        ):
            return "tfop_ready"

        if dc >= _DC_PLANET_HUNTERS:
            return "planet_hunters_discussion"

        return "github_only_reproducibility"

    if signal.mission in ("Kepler", "K2"):
        if (
            p_planet >= _KEPLER_PLANET_POSTERIOR
            and ns >= _KEPLER_NOVELTY
            and fpp <= _KEPLER_FPP
        ):
            return "kepler_archive_candidate"

        return "github_only_reproducibility"

    # ------------------------------------------------------------------
    # Mission-agnostic: paper / preprint or final fallback
    # (currently unreachable — see module docstring)
    # ------------------------------------------------------------------
    if (
        p_planet >= _PAPER_PLANET_POSTERIOR
        and ns >= _PAPER_NOVELTY
        and fpp <= _PAPER_FPP
        and (
            hi >= _PAPER_INTEREST
            or multi_planet_interest >= _PAPER_INTEREST
            or methodological_novelty >= _PAPER_INTEREST
        )
    ):
        return "paper_or_preprint_candidate"

    return "github_only_reproducibility"
