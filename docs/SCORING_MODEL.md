# SCORING MODEL

## Project
2026 Exoplanet Research

## Status
Research-grade specification v0.2

## Purpose
This document defines the candidate scoring framework for the TESS/Kepler exoplanet analysis toolkit.

The scoring system is intended to:
- Rank transit-like signals by credibility.
- Separate likely planets from false positives.
- Quantify uncertainty.
- Support reproducible submission decisions.
- Preserve scientific caution.

The model must never label an internally discovered signal as a confirmed planet. Confirmation status can only come from authoritative external catalogs.

---

# 1. Core Philosophy

Most apparent transit signals are not planets. The scoring system should therefore treat false positives as the default explanation until the signal survives multiple independent tests.

The system should favor:
- Interpretability over black-box prediction.
- Calibrated probabilities over arbitrary scores.
- Explicit false-positive explanations over single-number rankings.
- Conservative submission recommendations.
- Reproducibility over speed.

The initial implementation should be an interpretable Bayesian/logistic hybrid. Machine learning can be added later only after the pipeline has sufficient labeled validation data and calibration tests.

---

# 2. Candidate Hypotheses

For each signal, the scoring engine evaluates competing hypotheses.

## H_planet_candidate
The signal is consistent with a transiting planet candidate.

Evidence supporting this hypothesis:
- Periodic transit-like dips.
- Consistent depth across transits.
- Consistent duration across transits.
- No significant odd/even depth mismatch.
- No significant secondary eclipse.
- Physically plausible transit duration.
- Inferred planet radius is plausible.
- Centroid remains consistent with target.
- No known object match indicating false positive.

## H_eclipsing_binary
The signal is caused by an on-target eclipsing binary.

Evidence supporting this hypothesis:
- Odd/even transit depth mismatch.
- Secondary eclipse near phase 0.5.
- V-shaped eclipse morphology.
- Very deep transit-like event.
- Companion radius too large for a planet.
- Duration inconsistent with expected planetary transit.

## H_background_eclipsing_binary
The signal is caused by a blended nearby eclipsing binary.

Evidence supporting this hypothesis:
- Nearby bright Gaia/TIC source inside aperture.
- Transit-associated centroid shift.
- Difference image offset.
- High dilution or contamination risk.
- Signal appears stronger off-target than on-target.

## H_stellar_variability
The signal is caused by intrinsic stellar variability.

Evidence supporting this hypothesis:
- Quasi-periodic variability.
- Rotational modulation.
- Flare-like behavior.
- Broad non-box-shaped dips.
- Strong harmonics in the periodogram.
- Poor individual transit consistency.

## H_instrumental_artifact
The signal is caused by spacecraft/systematic effects.

Evidence supporting this hypothesis:
- Events coincide with quality flags.
- Events occur near sector boundaries.
- Events coincide with momentum dumps or background excursions.
- Similar signal appears in many nearby targets.
- Single-event detection with poor repeatability.

## H_known_object
The signal corresponds to an already known object.

Evidence supporting this hypothesis:
- Match to confirmed planet catalog.
- Match to TOI, KOI, CTOI, TCE, or known eclipsing binary.
- Period and epoch agree with cataloged event.

---

# 3. Required Outputs

Each candidate must produce these fields:

```json
{
  "candidate_id": "TIC_123456789_signal_01",
  "mission": "TESS",
  "target_id": "TIC 123456789",
  "period_days": 12.345,
  "epoch_bjd": 2459000.123,
  "duration_hours": 2.4,
  "depth_ppm": 850.0,
  "transit_count": 4,
  "snr": 11.2,

  "posterior": {
    "planet_candidate": 0.0,
    "eclipsing_binary": 0.0,
    "background_eclipsing_binary": 0.0,
    "stellar_variability": 0.0,
    "instrumental_artifact": 0.0,
    "known_object": 0.0
  },

  "scores": {
    "false_positive_probability": 0.0,
    "detection_confidence": 0.0,
    "novelty_score": 0.0,
    "habitability_interest": 0.0,
    "followup_value": 0.0,
    "submission_readiness": 0.0
  },

  "recommended_pathway": "github_only_reproducibility",
  "secondary_pathway": null,
  "explanation": {
    "positive_evidence": [],
    "negative_evidence": [],
    "blocking_issues": []
  }
}
```

The posterior probabilities should sum to approximately 1.0.

The false positive probability is:

```text
false_positive_probability =
    1 - posterior.planet_candidate
```

unless the candidate is already known, in which case it should be treated separately as a catalog annotation.

---

# 4. Bayesian Framework

The conceptual model is:

```text
P(H_i | D) = P(D | H_i) P(H_i) / Σ_j P(D | H_j) P(H_j)
```

Where:
- `H_i` is a candidate hypothesis.
- `D` is the observed data and diagnostics.
- `P(H_i)` is the prior probability of that hypothesis.
- `P(D | H_i)` is the likelihood of seeing the diagnostics under that hypothesis.

Because the early system will not have full generative likelihood models, implementation v1 should approximate the posterior with interpretable logit scores.

For each hypothesis:

```text
log_score_i = log_prior_i + weighted_evidence_i
posterior_i = softmax(log_score_i)
```

Where:

```text
softmax(log_score_i) = exp(log_score_i) / Σ_j exp(log_score_j)
```

---

# 5. First Implementation Strategy

## Use interpretable log-score models

Do not start with a black-box classifier.

For each hypothesis, define a transparent log score.

Example:

```text
log_score_planet =
    log_prior_planet
    + w_snr * log_snr_score
    + w_transits * transit_count_score
    + w_depth * depth_consistency_score
    + w_duration * duration_plausibility_score
    + w_shape * transit_shape_score
    - w_odd_even * odd_even_mismatch_score
    - w_secondary * secondary_eclipse_score
    - w_centroid * centroid_offset_score
    - w_contamination * contamination_score
    - w_systematics * systematics_overlap_score
```

Then compute normalized posterior probabilities across all hypotheses.

---

# 6. Feature Definitions

All feature scores should be normalized to `[0, 1]` unless otherwise stated.

## snr_score
Signal-to-noise score.

Suggested:

```text
snr_score = clip((snr - 5) / (12 - 5), 0, 1)
```

Interpretation:
- SNR < 5: weak.
- SNR 8: plausible.
- SNR 12+: strong.

## log_snr_score
More stable version of SNR.

```text
log_snr_score = clip(log(max(snr, 1)) / log(12), 0, 1)
```

## transit_count_score
Number of observed transits.

```text
1 transit  -> 0.25
2 transits -> 0.70
3+ transits -> 1.00
```

Single-transit signals should generally not be routed to formal submission without additional evidence.

## depth_consistency_score
Measures whether individual transit depths agree.

Possible implementation:
- Fit depth for each transit.
- Compute robust coefficient of variation.
- Convert to score.

```text
depth_consistency_score = 1 - clip(robust_cv_depth / threshold, 0, 1)
```

## duration_consistency_score
Measures whether transit durations are stable across events.

Same structure as depth consistency.

## duration_plausibility_score
Measures whether duration is plausible given:
- Period.
- Stellar radius.
- Stellar mass or density estimate.
- Circular orbit approximation.
- Allowed impact parameter range.

Score should be low if:
- Duration is implausibly long.
- Duration is too short relative to cadence/noise.
- Duration suggests stellar companion.

## odd_even_mismatch_score
Compares odd and even transit depths.

High value means likely eclipsing binary.

Possible implementation:
```text
odd_even_mismatch_score =
    clip(abs(depth_odd - depth_even) / sqrt(err_odd^2 + err_even^2) / 5, 0, 1)
```

Interpretation:
- 0: no mismatch.
- 1: highly significant mismatch.

## secondary_eclipse_score
Searches near phase 0.5 or expected occultation phases.

High value means likely eclipsing binary or self-luminous companion.

Possible implementation:
```text
secondary_eclipse_score = clip(secondary_snr / 7, 0, 1)
```

## transit_shape_score
Measures whether the event is box/U-shaped rather than V-shaped.

High value supports planet candidate.

Possible inputs:
- Ingress/egress fraction.
- Fitted impact parameter.
- V-shape metric.
- Transit model residuals.

## v_shape_score
High value supports eclipsing binary.

Approximately inverse of transit_shape_score.

## contamination_score
Measures risk from nearby sources.

Inputs:
- TIC contamination ratio.
- Gaia sources in aperture.
- Nearby stars within TESS pixel scale.
- Magnitude difference and angular separation.

High value means likely blended event.

## centroid_offset_score
Measures whether flux centroid shifts during transit.

High value supports background eclipsing binary.

## systematics_overlap_score
Measures overlap with known quality problems.

Inputs:
- TESS/Kepler quality flags.
- Momentum dump times.
- Sector boundary proximity.
- Background flux excursions.
- Common-mode signals in nearby targets.

## stellar_variability_score
Measures whether the light curve is better explained by stellar variability.

Inputs:
- Lomb-Scargle power at candidate period or harmonics.
- Autocorrelation.
- Rotational modulation strength.
- Flare event density.
- Transit duration relative to variability timescale.

## known_object_score
Probability that candidate is already cataloged.

Inputs:
- Period match.
- Epoch match.
- Sky-coordinate match.
- Target ID match.
- Catalog disposition.

If known object match is strong, route to `known_object_annotation`.

---

# 7. Initial Log-Score Model

These weights are starting points, not final scientific constants.

They should be calibrated later.

## Planet candidate score

```text
log_score_planet =
    log_prior_planet
    + 1.20 * log_snr_score
    + 1.00 * transit_count_score
    + 0.80 * depth_consistency_score
    + 0.70 * duration_consistency_score
    + 0.70 * duration_plausibility_score
    + 0.60 * transit_shape_score
    - 1.50 * odd_even_mismatch_score
    - 1.80 * secondary_eclipse_score
    - 1.40 * centroid_offset_score
    - 1.20 * contamination_score
    - 1.10 * systematics_overlap_score
    - 0.90 * stellar_variability_score
```

## Eclipsing binary score

```text
log_score_eclipsing_binary =
    log_prior_eb
    + 1.80 * odd_even_mismatch_score
    + 1.70 * secondary_eclipse_score
    + 1.40 * v_shape_score
    + 1.20 * large_depth_score
    + 1.20 * companion_radius_too_large_score
    + 0.80 * duration_implausibility_score
```

## Background eclipsing binary score

```text
log_score_background_eb =
    log_prior_background_eb
    + 1.80 * centroid_offset_score
    + 1.60 * contamination_score
    + 1.20 * nearby_bright_source_score
    + 1.00 * aperture_edge_score
    + 0.80 * dilution_sensitivity_score
```

## Stellar variability score

```text
log_score_stellar_variability =
    log_prior_stellar_variability
    + 1.50 * variability_periodogram_score
    + 1.20 * harmonic_score
    + 1.00 * flare_score
    + 1.00 * quasi_periodic_score
    + 0.80 * non_box_shape_score
```

## Instrumental artifact score

```text
log_score_instrumental =
    log_prior_instrumental
    + 1.70 * systematics_overlap_score
    + 1.30 * quality_flag_score
    + 1.20 * sector_boundary_score
    + 1.20 * background_excursion_score
    + 1.00 * single_event_score
    + 1.00 * nearby_targets_common_signal_score
```

## Known object score

```text
log_score_known_object =
    log_prior_known_object
    + 2.50 * target_id_match_score
    + 2.00 * period_match_score
    + 1.50 * epoch_match_score
    + 1.20 * coordinate_match_score
```

---

# 8. Priors

The first version can use conservative priors.

Suggested starting priors before normalization:

```text
P(planet_candidate)              = 0.10
P(eclipsing_binary)              = 0.20
P(background_eclipsing_binary)   = 0.20
P(stellar_variability)           = 0.20
P(instrumental_artifact)         = 0.20
P(known_object)                  = 0.10
```

These are intentionally pessimistic about new planet candidates.

Later priors should depend on:
- Stellar type.
- Galactic latitude/crowding.
- TESS magnitude.
- Aperture contamination.
- Observing baseline.
- Period/radius regime.
- Known occurrence rates.
- Pipeline detection completeness.

---

# 9. Derived Scores

## detection_confidence

Detection confidence asks:

> Is there a real, repeatable transit-like signal?

It should not answer whether the signal is planetary.

Suggested:

```text
detection_confidence =
    sigmoid(
        + 1.3 * log_snr_score
        + 1.1 * transit_count_score
        + 0.8 * depth_consistency_score
        + 0.7 * duration_consistency_score
        - 1.0 * systematics_overlap_score
        - 0.8 * data_gap_overlap_score
    )
```

## false_positive_probability

```text
false_positive_probability = 1 - posterior.planet_candidate
```

If `posterior.known_object` is high, report separately:

```text
catalog_status = known_object
```

## novelty_score

```text
novelty_score = 1 - known_object_score
```

Use conservative matching. If uncertain, mark as `possible_known_match`.

## habitability_interest

This is not a claim of habitability.

It means:

> Worth prioritizing for habitable-world follow-up.

Inputs:
- Estimated planet radius.
- Stellar effective temperature.
- Stellar radius.
- Stellar luminosity.
- Insolation estimate.
- Orbital period.
- Host brightness.
- Stellar activity risk.
- Ephemeris quality.

Suggested interpretation:
- High: small planet, temperate insolation, bright quiet host.
- Medium: possibly temperate or small but uncertain.
- Low: hot giant, poor ephemeris, noisy host, or weak signal.

## followup_value

Inputs:
- Host brightness.
- Transit depth.
- Transit duration.
- Ephemeris precision.
- Scientific interest.
- False positive probability.
- Observability.

Suggested:

```text
followup_value =
    0.30 * detection_confidence
    + 0.25 * novelty_score
    + 0.20 * habitability_interest
    + 0.15 * host_observability_score
    + 0.10 * ephemeris_quality_score
    - 0.30 * false_positive_probability
```

Clip to `[0, 1]`.

## submission_readiness

Measures whether the candidate is ready for external community attention.

```text
submission_readiness =
    0.35 * detection_confidence
    + 0.25 * posterior.planet_candidate
    + 0.15 * novelty_score
    + 0.15 * provenance_score
    + 0.10 * report_completeness_score
    - 0.25 * false_positive_probability
```

Clip to `[0, 1]`.

---

# 10. Submission Pathway Classification

The scoring model supports, but does not replace, conservative decision logic.

## Pathways

### known_object_annotation
Use when the signal matches a known catalog object.

### tfop_ready
Use for strong TESS candidates suitable for follow-up.

Minimum suggested criteria:
- mission = TESS
- not a known object
- transit_count >= 2, preferably 3+
- SNR >= 8
- posterior.planet_candidate >= 0.65
- false_positive_probability <= 0.35
- no strong secondary eclipse
- no significant odd/even mismatch
- contamination risk not high
- reproducible report available

### planet_hunters_discussion
Use for interesting but ambiguous TESS signals.

Examples:
- SNR between 5 and 8.
- Only 1–2 transits.
- Weird morphology needing human review.
- Possible signal but incomplete vetting.

### kepler_archive_candidate
Use for strong Kepler/K2 archival candidates.

Criteria:
- mission in Kepler/K2
- not known object
- repeated signal
- clean BLS peak
- reasonable vetting

### github_only_reproducibility
Use for weak, low-confidence, exploratory, or method-development findings.

### paper_or_preprint_candidate
Use only for unusually strong or scientifically exceptional cases.

Criteria:
- high posterior planet candidate probability
- low false positive probability
- strong novelty
- high scientific interest
- extensive vetting
- reproducible pipeline and report
- preferably collaborator review

---

# 11. Classification Pseudocode

```python
def classify_submission_pathway(c):
    if c.posterior.known_object >= 0.80:
        return "known_object_annotation"

    if c.false_positive_probability >= 0.70:
        return "github_only_reproducibility"

    if c.transit_count < 2:
        return "planet_hunters_discussion"

    if c.mission == "TESS":
        if (
            c.posterior.planet_candidate >= 0.65
            and c.false_positive_probability <= 0.35
            and c.snr >= 8
            and c.transit_count >= 2
            and c.contamination_score < 0.50
            and c.secondary_eclipse_score < 0.40
            and c.odd_even_mismatch_score < 0.40
            and c.provenance_score >= 0.80
        ):
            return "tfop_ready"

        if c.detection_confidence >= 0.45:
            return "planet_hunters_discussion"

        return "github_only_reproducibility"

    if c.mission in ["Kepler", "K2"]:
        if (
            c.posterior.planet_candidate >= 0.65
            and c.novelty_score >= 0.70
            and c.false_positive_probability <= 0.35
        ):
            return "kepler_archive_candidate"

        return "github_only_reproducibility"

    if (
        c.posterior.planet_candidate >= 0.80
        and c.novelty_score >= 0.80
        and c.false_positive_probability <= 0.20
        and (
            c.habitability_interest >= 0.70
            or c.multi_planet_interest >= 0.70
            or c.methodological_novelty >= 0.70
        )
    ):
        return "paper_or_preprint_candidate"

    return "github_only_reproducibility"
```

---

# 12. Calibration Plan

The initial model should not pretend to be calibrated.

Calibration is mandatory before probabilities are treated as meaningful.

## Validation sets

Use labeled examples from:
- Confirmed exoplanets.
- TOIs.
- KOIs.
- Kepler TCEs.
- Known false positives.
- Known eclipsing binaries.

## Calibration methods

Evaluate:
- Reliability curves.
- Precision-recall curves.
- ROC curves.
- Brier score.
- Confusion matrices by hypothesis type.
- Calibration by period/radius/SNR bins.

Potential calibration methods:
- Platt scaling.
- Isotonic regression.
- Bayesian logistic regression.
- Hierarchical calibration by mission.

## Desired property

If the model assigns `posterior.planet_candidate ≈ 0.80`, then roughly 80% of comparable validation examples should be planet-like under the chosen label definition.

---

# 13. Injection-Recovery Integration

Injection-recovery testing is required for survey-aware reliability.

## Purpose
Measure:
- Detection efficiency.
- False alarm behavior.
- Completeness.
- Sensitivity by planet radius and period.

## Process
1. Select real light curves.
2. Inject synthetic transit signals.
3. Run the full pipeline.
4. Measure recovery rate.
5. Map recovery as a function of:
   - planet radius
   - period
   - stellar type
   - noise level
   - observing baseline
   - transit count

## Completeness-aware adjustment

Use injection-recovery results to estimate:

```text
p_detectable_given_planet = completeness(radius, period, star, noise)
```

This should influence:
- detection confidence
- followup value
- interpretation of non-detections
- ranking of habitable-zone candidates

Do not over-penalize real candidates in difficult regions, but clearly label them as poorly calibrated if the pipeline has low recovery performance there.

---

# 14. Explanation Requirements

Every score must be accompanied by an explanation.

Example:

```json
{
  "positive_evidence": [
    "SNR = 11.2 exceeds strong-signal threshold",
    "4 observed transits with consistent depth",
    "No significant secondary eclipse detected"
  ],
  "negative_evidence": [
    "Nearby Gaia source within aperture increases contamination risk",
    "Only one TESS sector available"
  ],
  "blocking_issues": [
    "No centroid analysis available because target pixel file was not downloaded"
  ]
}
```

No candidate should be routed externally without a human-readable explanation.

---

# 15. Guardrails

The scoring engine must follow these rules:

- Never output “confirmed planet” for internal detections.
- Use “candidate signal,” “possible transit-like event,” or “follow-up target.”
- Always expose false-positive evidence.
- Always preserve provenance.
- Suppress formal submission if key diagnostics are missing.
- Prefer conservative classifications over optimistic ones.
- Store the scoring model version with every candidate output.
- Store all thresholds used for the run.
- Store raw and cleaned data provenance.

---

# 16. Recommended Implementation Files

Suggested package layout:

```text
src/exo_toolkit/
  scoring.py
  features.py
  hypotheses.py
  calibration.py
  pathway.py
  schemas.py
```

## schemas.py
Typed data models:
- CandidateSignal
- CandidateFeatures
- HypothesisPosterior
- CandidateScores
- PathwayRecommendation

## features.py
Feature extraction:
- SNR
- transit count
- odd/even mismatch
- secondary eclipse
- contamination
- systematics overlap

## hypotheses.py
Log-score models for each hypothesis.

## scoring.py
Posterior calculation and derived scores.

## calibration.py
Reliability curves, Platt scaling, isotonic regression.

## pathway.py
Submission pathway classification.

---

# 17. Versioning

Every candidate score must include:

```json
{
  "scoring_model_name": "bayesian_logscore_v0",
  "scoring_model_version": "0.2",
  "scoring_model_commit": "<git_commit_hash>",
  "threshold_config_hash": "<hash>"
}
```

This is required for reproducibility.

---

# 18. Open Questions

These are intentionally unresolved and should become future decisions in `DECISIONS.md`.

1. Should the first BLS search be supplemented with Transit Least Squares?
2. What minimum SNR should define a candidate in v1?
3. Should priors be mission-specific from the beginning?
4. Should known-object matching be mandatory before scoring or after scoring?
5. How should single-transit events be represented?
6. Should habitability interest use optimistic or conservative habitable-zone limits in v1?
7. Should the scoring model be trained separately for TESS and Kepler?

---

# 19. Immediate Implementation Target

The first implementation should support:

```text
Input:
- Candidate period
- Epoch
- Duration
- Depth
- SNR
- Transit count
- Basic vetting metrics

Output:
- posterior probabilities
- false positive probability
- detection confidence
- recommended pathway
- explanation
```

Do not wait for perfect astrophysics. Build a transparent v0 model, then calibrate it.

---

# 20. Summary

The scoring model is the intellectual core of this project.

It should answer five questions:

1. Is the signal real?
2. Is it likely planetary?
3. What false-positive explanation is most plausible?
4. Is it scientifically interesting?
5. Where should it go next?

The first version should be conservative, interpretable, reproducible, and easy to improve.
