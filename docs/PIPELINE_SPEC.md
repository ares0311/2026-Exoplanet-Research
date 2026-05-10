# PIPELINE SPECIFICATION

## Purpose

Define a **deterministic, reproducible pipeline** for exoplanet candidate detection.

---

## High-Level Flow

```
Fetch → Clean → Search → Vet → Score → Classify
```

Each stage must:
- produce outputs to disk
- be independently testable
- preserve provenance

---

## 1. Fetch

### Input
- Target ID / coordinates

### Process
- Query MAST via Lightkurve
- Download light curves and metadata

### Output
- Raw FITS files
- Metadata JSON

---

## 2. Clean

### Goals
- Remove noise/systematics
- Preserve transit signals

### Steps
- Remove NaNs
- Sigma clipping
- Normalize flux
- Detrend (windowed filter)

### Output
- Cleaned light curve (parquet/csv)

---

## 3. Search

### Algorithm
- Box Least Squares (BLS)

### Outputs
- Period
- Epoch
- Duration
- Depth
- SNR

### Requirements
- Multi-peak detection
- Iterative masking for multi-planet systems

---

## 4. Vet

### Tests
- Odd/even depth comparison
- Secondary eclipse detection
- Duration plausibility
- Contamination check
- Systematics overlap

### Output
- Vetting metrics

---

## 5. Score

### Method
- Bayesian hypothesis model

### Outputs
- Posterior probabilities
- False positive probability
- Detection confidence

---

## 6. Classify

### Pathways
- TFOP-ready
- Planet Hunters
- Archive candidate
- GitHub-only
- Known object

### Decision Logic
- Based on scoring thresholds
- Conservative by design

---

## Data Contracts

Each stage must output:
- structured data
- metadata
- reproducibility info

---

## Non-Goals (v1)

- No ML-first approach
- No real-time processing
- No confirmation claims

---

## Design Principles

- Reproducibility first
- Transparency over performance
- Conservative classification
- Modular architecture
- Local runtime sizing should follow `docs/SYSTEM_PROFILE.md` while remaining configurable and portable
