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

---

## 3. Methodology

### 3.1 Transit Photometry Fundamentals

A transiting planet occults a fraction of its host star's disk, producing a periodic reduction in observed flux. For a planet of radius $R_p$ orbiting a star of radius $R_\star$, the fractional transit depth is

$$\delta = \left(\frac{R_p}{R_\star}\right)^2.$$

For a circular orbit with semi-major axis $a$, impact parameter $b = (a/R_\star)\cos i$, and ratio $k = R_p/R_\star$, the total transit duration (first to fourth contact) is

$$T_{14} = \frac{P}{\pi} \arcsin\!\left(\frac{R_\star}{a} \sqrt{(1 + k)^2 - b^2}\right),$$

where $P$ is the orbital period (Winn, 2010). The ingress/egress duration is

$$T_{12} = \frac{P}{\pi} \arcsin\!\left(\frac{R_\star}{a} \sqrt{(1 - k)^2 - b^2}\right).$$

A box-shaped (flat-bottomed) transit satisfies $T_{12} \ll T_{14}$, indicative of a small planet-to-star radius ratio and/or low impact parameter. The ratio $T_{12}/T_{14}$ is used in the pipeline as the transit shape diagnostic `ingress_egress_fraction`.

### 3.2 Box Least Squares Periodicity Search

Transit candidates are identified using the Box Least Squares algorithm of Kovács et al. (2002), as implemented in `astropy.timeseries.BoxLeastSquares`. For a light curve with $N$ cadences $(t_i, f_i, \sigma_i)$, define the inverse-variance weights $w_i = \sigma_i^{-2}$. For trial period $P$, reference epoch $t_0$, and fractional duration $q$, the phase of each observation is

$$\phi_i = \frac{(t_i - t_0) \bmod P}{P} \in [0, 1).$$

The in-transit index set is $\mathcal{T}(P, t_0, q) = \{i : \phi_i \leq q\}$ and the out-of-transit set is $\mathcal{O} = \{1,\ldots,N\} \setminus \mathcal{T}$. The weighted mean fluxes are

$$\bar{f}_{\mathcal{T}} = \frac{\sum_{i \in \mathcal{T}} w_i f_i}{\sum_{i \in \mathcal{T}} w_i}, \qquad \bar{f}_{\mathcal{O}} = \frac{\sum_{i \in \mathcal{O}} w_i f_i}{\sum_{i \in \mathcal{O}} w_i},$$

and the depth estimate is $\hat{s} = \bar{f}_{\mathcal{O}} - \bar{f}_{\mathcal{T}}$. The BLS power spectrum is evaluated over a grid of $(P, t_0, q)$ triples and the Signal Detection Efficiency is

$$\mathrm{SDE}(P) = \frac{\hat{s}(P) - \langle \hat{s} \rangle}{\mathrm{std}(\hat{s})},$$

where the mean and standard deviation are taken over all trial periods at fixed best $(t_0, q)$.

The duration grid is capped at $q_{\max} = 0.9\, P_{\min} / 24$ hours to satisfy the strict BLS constraint that the maximum trial duration must be shorter than the minimum trial period (Kovács et al., 2002). Multi-planet candidates are recovered by iterative transit masking: after each BLS peak the corresponding in-transit cadences are masked and the search is repeated on the residual series.

### 3.3 Bayesian Multi-Hypothesis Scoring

#### 3.3.1 Competing Hypotheses

For each detected signal $\mathbf{D}$, the scoring engine evaluates six mutually exclusive hypotheses $H_i$:

| Symbol | Hypothesis | Prior $P(H_i)$ |
|---|---|---|
| $H_\mathrm{pc}$ | Planet candidate | 0.10 |
| $H_\mathrm{eb}$ | On-target eclipsing binary | 0.20 |
| $H_\mathrm{beb}$ | Background eclipsing binary | 0.20 |
| $H_\mathrm{sv}$ | Stellar variability | 0.20 |
| $H_\mathrm{ia}$ | Instrumental artifact | 0.20 |
| $H_\mathrm{ko}$ | Known catalog object | 0.10 |

Priors are intentionally pessimistic regarding new planet candidates, consistent with the empirical false-positive rates reported by Fressin et al. (2013) and Morton (2012).

#### 3.3.2 Log-Score Approximation

Because full generative likelihood models $P(\mathbf{D} \mid H_i)$ require detailed stellar and instrumental forward models not available in the early pipeline, the posterior is approximated via a log-score model (Díaz et al., 2014):

$$\ell_i = \log P(H_i) + \sum_{k} w_{ik}\, \phi_k(\mathbf{D}),$$

where $\phi_k(\mathbf{D}) \in [0, 1]$ are normalized diagnostic feature scores and $w_{ik} \in \mathbb{R}$ are hypothesis-specific weights. Positive weights contribute evidence for $H_i$; negative weights contribute evidence against it. Features that are unavailable (e.g., because a catalog query was not performed) are assigned $\phi_k = 0$, contributing neither evidence for nor against any hypothesis.

The normalized posterior is computed via a numerically stable softmax. Letting $\ell_{\max} = \max_i \ell_i$:

$$p_i = P(H_i \mid \mathbf{D}) \approx \frac{\exp(\ell_i - \ell_{\max})}{\displaystyle\sum_{j=1}^{6} \exp(\ell_j - \ell_{\max})}.$$

Subtracting $\ell_{\max}$ before exponentiation prevents floating-point overflow without affecting the normalized result.

#### 3.3.3 False Positive Probability

The false positive probability is defined as

$$\mathrm{FPP} = 1 - p_{\mathrm{pc}}.$$

When $p_{\mathrm{ko}} \geq 0.80$, the signal is reclassified as a known-object annotation and FPP is reported separately to avoid conflating genuine false positives with catalog matches.

### 3.4 Diagnostic Feature Extraction

All features $\phi_k : \mathbf{D} \to [0, 1]$ are typed as `OptScore = float | None`. A `None` value indicates the diagnostic was not computed (missing data or insufficient coverage) and contributes zero to all log scores.

**Signal-to-noise score.**  Using a logarithmic transform for numerical stability across the wide dynamic range of survey SNR values:

$$\phi_{\mathrm{SNR}} = \mathrm{clip}\!\left(\frac{\log\max(\mathrm{SNR},\, 1)}{\log 12},\; 0,\; 1\right).$$

**Transit count score.**  Discrete mapping reflecting the requirement for repeated events:

$$\phi_{\mathrm{tc}} = \begin{cases} 0.25 & n_\mathrm{transits} = 1 \\ 0.70 & n_\mathrm{transits} = 2 \\ 1.00 & n_\mathrm{transits} \geq 3 \end{cases}$$

**Odd/even depth mismatch.**  An eclipsing binary with unequal primary and secondary depths will produce alternating transit depths when phase-folded at half the true period. The mismatch significance is

$$\phi_{\mathrm{OE}} = \mathrm{clip}\!\left(\frac{|\delta_\mathrm{odd} - \delta_\mathrm{even}|}{5\,\sqrt{\sigma_\mathrm{odd}^2 + \sigma_\mathrm{even}^2}},\; 0,\; 1\right),$$

where $\delta_\mathrm{odd}$, $\delta_\mathrm{even}$ are the inverse-variance-weighted mean depths of odd- and even-numbered transits (in ppm) and $\sigma$ are their propagated uncertainties.

**Secondary eclipse SNR.**  The pipeline searches for a secondary eclipse at orbital phase $\phi = 0.5$ (superior conjunction, as expected for a circular orbit):

$$\phi_{\mathrm{sec}} = \mathrm{clip}\!\left(\frac{\mathrm{SNR}_{\phi=0.5}}{7},\; 0,\; 1\right),$$

where $\mathrm{SNR}_{\phi=0.5}$ is the depth divided by its propagated uncertainty in the secondary window, estimated relative to the out-of-transit baseline.

**Transit shape (ingress/egress fraction).**  Comparing the mean depth in the outer half of the transit window ($T/4 < |t - t_c| \leq T/2$) to the inner half ($|t - t_c| \leq T/4$):

$$\phi_{\mathrm{shape}} = \mathrm{clip}\!\left(\frac{\bar{\delta}_{\mathrm{outer}}}{\bar{\delta}_{\mathrm{inner}}},\; 0,\; 1\right).$$

Values near 1 indicate a flat-bottomed (box-shaped) transit consistent with a planet; values near 0 indicate a V-shaped morphology consistent with a grazing eclipse.

**Data gap fraction.**  The fraction of expected transit windows that contain fewer than three in-transit cadences:

$$\phi_{\mathrm{gap}} = \frac{|\{n : |{\{i : |t_i - t_n| \leq T/2\}}| < 3\}|}{N_{\mathrm{windows}}},$$

where $t_n = t_0 + nP$ are the predicted transit centers. This score enters as a penalty on detection confidence.

### 3.5 Posterior Calibration

The initial log-score model is not calibrated by construction. Calibration maps raw model probabilities to empirical frequencies using labeled examples drawn from confirmed planets, TOIs, known false positives, and eclipsing binary catalogs.

#### 3.5.1 Brier Score

Model reliability is quantified per hypothesis using the Brier score (Brier, 1950):

$$\mathrm{BS}_k = \frac{1}{N} \sum_{i=1}^{N} \left(p_{ik} - y_{ik}\right)^2,$$

where $y_{ik} \in \{0, 1\}$ is the binary true-label indicator for hypothesis $k$ in sample $i$, and $p_{ik}$ is the model's posterior probability. A perfectly calibrated model achieves $\mathrm{BS}_k = 0$; the naive uniform prior achieves $\mathrm{BS}_k = 5/6 \cdot (1/6)^2 + 1/6 \cdot (5/6)^2 \approx 0.139$.

#### 3.5.2 Platt Scaling

Platt scaling (Platt, 1999) fits a logistic sigmoid to the raw model probabilities in a one-vs-rest framework. For hypothesis $k$, the calibrated probability is

$$\tilde{p}_k = \sigma(a_k p_k + b_k), \qquad \sigma(x) = \frac{1}{1 + e^{-x}},$$

where parameters $(a_k, b_k)$ are found by minimizing the negative log-likelihood of the sigmoid over the training labels via the Nelder-Mead simplex method (Scipy). The identity mapping $(a_k, b_k) = (1, 0)$ is used as a fallback when fewer than five training samples are available or when all labels are identical.

#### 3.5.3 Isotonic Regression (PAVA)

Isotonic regression (Barlow et al., 1972) provides a non-parametric calibration mapping by fitting the largest monotone non-decreasing step function to the empirical $(p_k, y_k)$ pairs. The Pool Adjacent Violators Algorithm (PAVA) solves

$$\min_{\tilde{p}} \sum_{i=1}^{N} \left(p_i - \tilde{p}_i\right)^2 \quad \text{subject to} \quad \tilde{p}_1 \leq \tilde{p}_2 \leq \cdots \leq \tilde{p}_N,$$

in $O(N)$ time by merging blocks of adjacent violating values into their pooled mean. The pipeline implements PAVA in pure Python without any dependency on scikit-learn.

After applying either calibration method in a one-vs-rest fashion, the six calibrated probabilities are renormalized to sum to unity before constructing the final `HypothesisPosterior`.

### 3.6 Submission Pathway Classification

Candidates are routed to one of six submission pathways by an ordered decision gate evaluated in strict sequence:

| Priority | Condition | Pathway |
|---|---|---|
| 1 | $p_{\mathrm{ko}} \geq 0.80$ | `known_object_annotation` |
| 2 | $\mathrm{FPP} \geq 0.70$ | `github_only_reproducibility` |
| 3 | $n_{\mathrm{transits}} < 2$ | `planet_hunters_discussion` |
| 4 | TESS + all 9 quality gates pass | `tfop_ready` |
| 4 | TESS + detection confidence $\geq 0.45$ | `planet_hunters_discussion` |
| 5 | Kepler/K2 + $p_{\mathrm{pc}} \geq 0.65$, novelty $\geq 0.70$, FPP $\leq 0.35$ | `kepler_archive_candidate` |
| 6 | Fallback | `github_only_reproducibility` |

The nine conditions for `tfop_ready` include: $\mathrm{SNR} \geq 8$, $n_{\mathrm{transits}} \geq 2$, $p_{\mathrm{pc}} \geq 0.65$, $\mathrm{FPP} \leq 0.35$, contamination score $< 0.50$, secondary eclipse score $< 0.40$, odd/even mismatch score $< 0.40$, no known-object match, and a valid provenance score. A `None` feature score conservatively fails any gate condition it participates in.

