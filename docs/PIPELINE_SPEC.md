# PIPELINE SPECIFICATION

## Overview
End-to-end exoplanet candidate detection system.

## Pipeline Stages

1. Fetch
   - Source: TESS/Kepler via Lightkurve
   - Output: raw light curve

2. Clean
   - Remove NaNs/outliers
   - Normalize flux
   - Flatten trends

3. Search
   - Algorithm: BLS
   - Output: candidate signals

4. Vet
   - Odd/even test
   - Secondary eclipse detection
   - Contamination checks

5. Score
   - Bayesian hypothesis model
   - Output probabilities

6. Classify
   - Assign submission pathway

---

## Outputs
- Candidate list
- Probabilities
- Reports
