# 2026 Exoplanet Research

[![CI](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-406%20passing-brightgreen.svg)](tests/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Abstract

This repository implements a complete, reproducible computational pipeline for the detection, vetting, and probabilistic classification of exoplanet transit candidates in photometric time-series data from the Transiting Exoplanet Survey Satellite (TESS) and the Kepler/K2 missions. The pipeline proceeds through six deterministic stages — data acquisition, preprocessing, Box Least Squares (BLS) periodicity search, signal vetting, Bayesian multi-hypothesis scoring, and submission pathway classification — and outputs calibrated posterior probabilities over six competing astrophysical and instrumental hypotheses. A conservative log-score approximation to Bayes' theorem is employed in lieu of generative likelihood models, with posterior calibration implemented via Platt scaling and isotonic regression (Pool Adjacent Violators Algorithm). The system is designed around scientific caution: it never labels an internally detected signal as a confirmed planet, exposes all false-positive evidence alongside each candidate score, and defers to authoritative external catalogs for confirmation status. The complete implementation comprises ten Python modules, 406 unit and integration tests, strict static typing (mypy), and continuous integration via GitHub Actions.

---

## 1. Introduction

The detection of transiting exoplanets from space-based photometry has undergone a paradigm shift from individual targeted observations toward large-scale automated surveys. The Kepler mission (Borucki et al., 2010) surveyed approximately 150,000 stars continuously for four years, yielding more than 4,000 planet candidates and establishing the statistical framework for occurrence-rate studies (Fressin et al., 2013; Bryson et al., 2021). Its successor, the Transiting Exoplanet Survey Satellite (Ricker et al., 2015), observes nearly the entire sky in 27-day sectors, generating a continuous stream of TESS Objects of Interest (TOIs) that require community vetting before resources are allocated for ground-based follow-up.

A persistent challenge across both missions is the high false-positive rate among photometric transit candidates. Background eclipsing binaries, on-target eclipsing binaries diluted by the target's flux, stellar variability masquerading as periodic dimming, and instrumental systematics collectively account for the majority of transit-like signals detected by automated pipelines (Fressin et al., 2013; Morton, 2012). Rigorous vetting — combining photometric diagnostics, centroid analysis, catalog matching, and probabilistic modeling — is therefore a prerequisite for responsible candidate reporting.

Citizen-science initiatives such as Planet Hunters have demonstrated that human inspection of phase-folded light curves can recover candidates missed by automated pipelines, particularly single-transit events and long-period systems (Fischer et al., 2012). However, the volume of data produced by TESS renders manual inspection alone insufficient. A computational toolkit that automates the vetting and scoring workflow, while remaining interpretable and reproducible, occupies a productive niche between fully automated survey pipelines and ad hoc visual inspection.

This project addresses that niche. The `exo_toolkit` Python package implements a fully automated, end-to-end pipeline from raw MAST data retrieval through calibrated candidate scoring and submission pathway routing. Every scoring decision is accompanied by a structured explanation enumerating positive evidence, negative evidence, and blocking issues. All intermediate results preserve full provenance. The system targets lightly-worked TESS targets — later sectors, fainter stars (TESS magnitude 10–14), and less-crowded fields — where the probability of genuine novel discoveries is highest relative to the existing literature.

---

## 2. Pipeline Architecture

The pipeline is organized as six sequential, independently testable stages:

```
┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────┐   ┌───────┐   ┌──────────┐
│  Fetch  │──▶│  Clean  │──▶│  Search  │──▶│ Vet │──▶│ Score │──▶│ Classify │
└─────────┘   └─────────┘   └──────────┘   └─────┘   └───────┘   └──────────┘
```

Each stage produces a typed, immutable result object and preserves provenance metadata. Stages communicate exclusively through these result objects; there is no shared mutable state.

| Module | Stage | Responsibility | Tests |
|---|---|---|---|
| `fetch.py` | Fetch | MAST data retrieval via Lightkurve; records cadence, sectors, pipeline | 40 (+2 live) |
| `clean.py` | Clean | NaN removal, sigma clipping, normalization, windowed detrending | 39 |
| `search.py` | Search | BLS periodicity search; iterative transit masking for multi-planet systems | 43 |
| `vet.py` | Vet | Per-transit depth measurement, odd/even comparison, secondary eclipse, transit shape, data-gap fraction | 47 |
| `scoring.py` | Score | Log-score posterior computation; derived scores (FPP, detection confidence, novelty) | 25 |
| `pathway.py` | Classify | Ordered gate-based submission pathway classification | 35 |
| `schemas.py` | — | Immutable Pydantic data models for all pipeline types | 33 |
| `features.py` | — | `RawDiagnostics` container; `extract_features()` mapping diagnostics to `[0,1]` scores | 89 |
| `hypotheses.py` | — | Per-hypothesis log-score functions | 28 |
| `calibration.py` | — | Platt scaling, PAVA isotonic regression, Brier score, reliability curves | 70 |
| **Total** | | | **406** |

