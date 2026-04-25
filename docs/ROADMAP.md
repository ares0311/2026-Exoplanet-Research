# ROADMAP

## Milestone 1 — Minimal Detection Pipeline
- Fetch TESS light curve
- Clean and normalize
- Run BLS
- Output candidates

## Milestone 2 — Vetting Layer
- Odd/even depth test
- Secondary eclipse detection
- Duration plausibility

## Milestone 3 — Bayesian Scoring Engine
- Hypothesis modeling
- Posterior probabilities
- False positive classification

## Milestone 4 — Submission Classification
- TFOP-ready
- Zooniverse discussion
- Archive candidates

## Milestone 5 — Reporting
- Plots
- Markdown/HTML reports

---

## Decision Tree

IF known_object:
    → known_object_annotation

ELIF p_planet > 0.65 AND SNR > 8:
    → tfop_ready

ELIF ambiguous:
    → planet_hunters_discussion

ELSE:
    → github_only
