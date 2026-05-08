# 2026 Exoplanet Research

[![CI](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-696%20passing-brightgreen.svg)](tests/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Abstract

This repository implements a complete, reproducible computational pipeline for the detection, vetting, and probabilistic classification of exoplanet transit candidates in photometric time-series data from the Transiting Exoplanet Survey Satellite (TESS) and the Kepler/K2 missions. The pipeline proceeds through six deterministic stages — data acquisition, preprocessing, Box Least Squares (BLS) periodicity search, signal vetting, Bayesian multi-hypothesis scoring, and submission pathway classification — and outputs calibrated posterior probabilities over six competing astrophysical and instrumental hypotheses. A conservative log-score approximation to Bayes' theorem is employed in lieu of generative likelihood models, with posterior calibration implemented via Platt scaling and isotonic regression (Pool Adjacent Violators Algorithm). An optional Tier-1 XGBoost classifier and Tier-3 stacking meta-learner augment the Bayesian scorer when labelled training data are available. The system is designed around scientific caution: it never labels an internally detected signal as a confirmed planet, exposes all false-positive evidence alongside each candidate score, and defers to authoritative external catalogs for confirmation status. The complete implementation comprises thirteen Python modules, 696 unit and integration tests, strict static typing (mypy), and continuous integration via GitHub Actions.

---

## 1. Introduction

The detection of transiting exoplanets from space-based photometry has undergone a paradigm shift from individual targeted observations toward large-scale automated surveys. The Kepler mission (Borucki et al.) surveyed approximately 150,000 stars continuously for four years, yielding more than 4,000 planet candidates and establishing the statistical framework for occurrence-rate studies (Fressin et al.; Bryson et al.). Its successor, the Transiting Exoplanet Survey Satellite (Ricker et al.), observes nearly the entire sky in 27-day sectors, generating a continuous stream of TESS Objects of Interest (TOIs) that require community vetting before resources are allocated for ground-based follow-up (Guerrero et al.).

A persistent challenge across both missions is the high false-positive rate among photometric transit candidates. Background eclipsing binaries, on-target eclipsing binaries diluted by the target's flux, stellar variability masquerading as periodic dimming, and instrumental systematics collectively account for the majority of transit-like signals detected by automated pipelines (Fressin et al.; Morton). Rigorous vetting — combining photometric diagnostics, centroid analysis, catalog matching, and probabilistic modeling — is therefore a prerequisite for responsible candidate reporting. The Kepler mission's Robovetter system (Coughlin et al.; Thompson et al.) demonstrated that automated multi-criterion vetting can achieve high completeness and reliability when trained on a large labeled corpus; the same principles apply to TESS data, with appropriate corrections for differences in cadence, pixel scale, and systematic noise.

Deep-learning approaches have extended the automated vetting paradigm further. Shallue and Vanderburg showed that a convolutional neural network trained on Kepler TCEs can match or exceed human performance on the classification of folded light curves, recovering an eighth planet in the Kepler-90 system. However, such models are mission-specific and require thousands of labeled examples before they generalize reliably (Shallue and Vanderburg). This pipeline therefore implements the CNN as a gated Tier-2 component that is withheld until 5,000 or more TESS confirmed-planet labels become available, using the Bayesian log-score model and XGBoost tabular classifier as production-ready fallbacks in the interim.

Citizen-science initiatives such as Planet Hunters have demonstrated that human inspection of phase-folded light curves can recover candidates missed by automated pipelines, particularly single-transit events and long-period systems (Fischer et al.). However, the volume of data produced by TESS renders manual inspection alone insufficient. A computational toolkit that automates the vetting and scoring workflow, while remaining interpretable and reproducible, occupies a productive niche between fully automated survey pipelines and ad hoc visual inspection.

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
| `cli.py` | — | `exo <TIC-ID>` entry point; `--scorer`, `--model-path`, `--output` options | 20 |
| `ml/xgboost_scorer.py` | — | XGBoost tabular classifier on 35 `OptScore` features | 45 |
| `ml/stacking_scorer.py` | — | Weighted blend of XGBoost + Bayesian posteriors | 22 |
| **Total** | | | **696** |

### ML Scoring Modes

The `--scorer` flag selects among three backends:

| Mode | Flag | Description |
|---|---|---|
| Bayesian (default) | `--scorer bayesian` | Log-score softmax posterior; no labels required |
| XGBoost Tier-1 | `--scorer xgboost --model-path model.json` | Tabular classifier on 35 feature scores |
| Ensemble Tier-3 | `--scorer ensemble --model-path model.json` | Weighted blend `w·P_xgb + (1-w)·P_bayes` |

The Tier-2 CNN (1D convolution on phase-folded flux) is gated on 5,000+ TESS confirmed-planet labels. Check the current label count:

```bash
python Skills/count_tess_labels.py
```

---

## 3. Methodology

### 3.1 Transit Photometry Fundamentals

A transiting planet occults a fraction of its host star's disk, producing a periodic reduction in observed flux. For a planet of radius $R_p$ orbiting a star of radius $R_\star$, the fractional transit depth is

$$\delta = \left(\frac{R_p}{R_\star}\right)^2.$$

For a circular orbit with semi-major axis $a$, impact parameter $b = (a/R_\star)\cos i$, and ratio $k = R_p/R_\star$, the total transit duration (first to fourth contact) is

$$T_{14} = \frac{P}{\pi} \arcsin\!\left(\frac{R_\star}{a} \sqrt{(1 + k)^2 - b^2}\right),$$

where $P$ is the orbital period (Winn). The ingress/egress duration is

$$T_{12} = \frac{P}{\pi} \arcsin\!\left(\frac{R_\star}{a} \sqrt{(1 - k)^2 - b^2}\right).$$

A box-shaped (flat-bottomed) transit satisfies $T_{12} \ll T_{14}$, indicative of a small planet-to-star radius ratio and/or low impact parameter. The ratio $T_{12}/T_{14}$ is used in the pipeline as the transit shape diagnostic `ingress_egress_fraction`.

### 3.2 Box Least Squares Periodicity Search

Transit candidates are identified using the Box Least Squares algorithm of Kovács et al., as implemented in `astropy.timeseries.BoxLeastSquares`. For a light curve with $N$ cadences $(t_i, f_i, \sigma_i)$, define the inverse-variance weights $w_i = \sigma_i^{-2}$. For trial period $P$, reference epoch $t_0$, and fractional duration $q$, the phase of each observation is

$$\phi_i = \frac{(t_i - t_0) \bmod P}{P} \in [0, 1).$$

The in-transit index set is $\mathcal{T}(P, t_0, q) = \{i : \phi_i \leq q\}$ and the out-of-transit set is $\mathcal{O} = \{1,\ldots,N\} \setminus \mathcal{T}$. The weighted mean fluxes are

$$\bar{f}_{\mathcal{T}} = \frac{\sum_{i \in \mathcal{T}} w_i f_i}{\sum_{i \in \mathcal{T}} w_i}, \qquad \bar{f}_{\mathcal{O}} = \frac{\sum_{i \in \mathcal{O}} w_i f_i}{\sum_{i \in \mathcal{O}} w_i},$$

and the depth estimate is $\hat{s} = \bar{f}_{\mathcal{O}} - \bar{f}_{\mathcal{T}}$. The BLS power spectrum is evaluated over a grid of $(P, t_0, q)$ triples and the Signal Detection Efficiency is

$$\mathrm{SDE}(P) = \frac{\hat{s}(P) - \langle \hat{s} \rangle}{\mathrm{std}(\hat{s})},$$

where the mean and standard deviation are taken over all trial periods at fixed best $(t_0, q)$.

The duration grid is capped at $q_{\max} = 0.9\, P_{\min} / 24$ hours to satisfy the strict BLS constraint that the maximum trial duration must be shorter than the minimum trial period (Kovács et al.). Multi-planet candidates are recovered by iterative transit masking: after each BLS peak the corresponding in-transit cadences are masked and the search is repeated on the residual series.

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

Priors are intentionally pessimistic regarding new planet candidates, consistent with the empirical false-positive rates reported by Fressin et al. and Morton.

#### 3.3.2 Log-Score Approximation

Because full generative likelihood models $P(\mathbf{D} \mid H_i)$ require detailed stellar and instrumental forward models not available in the early pipeline, the posterior is approximated via a log-score model (Díaz et al.):

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

$$\phi_{\mathrm{gap}} = \frac{|\{n : |\{i : |t_i - t_n| \leq T/2\}| < 3\}|}{N_{\mathrm{windows}}},$$

where $t_n = t_0 + nP$ are the predicted transit centers. This score enters as a penalty on detection confidence.

### 3.5 ML Scorer — XGBoost Tier-1

When labelled training data are available the Bayesian scorer can be replaced or augmented by a gradient-boosted tree classifier. The XGBoost model takes as input the vector of 35 `OptScore` fields from `CandidateFeatures`, with `None` values passed as `NaN` and handled natively by XGBoost's missing-value splitting. The model is trained using stratified k-fold cross-validation (Coughlin et al.; Thompson et al.) on the Kepler DR25 cumulative planet candidate table or the TESS TOI disposition table, labelling confirmed planets as positive and false positives / eclipsing binaries as negative.

Out-of-fold predicted probabilities are collected during cross-validation to fit Platt scaling parameters $(a, b)$ by minimizing the binary cross-entropy:

$$\mathcal{L}(a, b) = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \sigma(a p_i + b) + (1-y_i)\log(1-\sigma(a p_i + b))\right],$$

where $\sigma(x) = (1 + e^{-x})^{-1}$ and $p_i$ are the raw XGBoost probability outputs. The parameters $(a, b)$ are stored alongside the model weights in the metadata JSON and applied automatically at inference time.

### 3.6 Posterior Calibration

The initial log-score model is not calibrated by construction. Calibration maps raw model probabilities to empirical frequencies using labeled examples drawn from confirmed planets, TOIs, known false positives, and eclipsing binary catalogs.

#### 3.6.1 Brier Score

Model reliability is quantified per hypothesis using the Brier score (Brier):

$$\mathrm{BS}_k = \frac{1}{N} \sum_{i=1}^{N} \left(p_{ik} - y_{ik}\right)^2,$$

where $y_{ik} \in \{0, 1\}$ is the binary true-label indicator for hypothesis $k$ in sample $i$, and $p_{ik}$ is the model's posterior probability. A perfectly calibrated model achieves $\mathrm{BS}_k = 0$; the naive uniform prior achieves $\mathrm{BS}_k = 5/6 \cdot (1/6)^2 + 1/6 \cdot (5/6)^2 \approx 0.139$.

#### 3.6.2 Platt Scaling

Platt scaling (Platt) fits a logistic sigmoid to the raw model probabilities in a one-vs-rest framework. For hypothesis $k$, the calibrated probability is

$$\tilde{p}_k = \sigma(a_k p_k + b_k), \qquad \sigma(x) = \frac{1}{1 + e^{-x}},$$

where parameters $(a_k, b_k)$ are found by minimizing the negative log-likelihood of the sigmoid over the training labels via the Nelder-Mead simplex method (Scipy). The identity mapping $(a_k, b_k) = (1, 0)$ is used as a fallback when fewer than five training samples are available or when all labels are identical.

#### 3.6.3 Isotonic Regression (PAVA)

Isotonic regression (Barlow et al.) provides a non-parametric calibration mapping by fitting the largest monotone non-decreasing step function to the empirical $(p_k, y_k)$ pairs. The Pool Adjacent Violators Algorithm (PAVA) solves

$$\min_{\tilde{p}} \sum_{i=1}^{N} \left(p_i - \tilde{p}_i\right)^2 \quad \text{subject to} \quad \tilde{p}_1 \leq \tilde{p}_2 \leq \cdots \leq \tilde{p}_N,$$

in $O(N)$ time by merging blocks of adjacent violating values into their pooled mean. The pipeline implements PAVA in pure Python without any dependency on scikit-learn.

After applying either calibration method in a one-vs-rest fashion, the six calibrated probabilities are renormalized to sum to unity before constructing the final `HypothesisPosterior`.

### 3.7 Submission Pathway Classification

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

---

## 4. Installation

The package is not yet published to PyPI. Install directly from source:

```bash
git clone https://github.com/ares0311/2026-Exoplanet-Research.git
cd 2026-Exoplanet-Research
pip install -r requirements.txt
```

The package is not installed into site-packages; use `PYTHONPATH=src` when running scripts or tests:

```bash
export PYTHONPATH=src
python -c "from exo_toolkit.fetch import fetch_lightcurve; print('OK')"
```

Alternatively, install in editable mode:

```bash
pip install -e .
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations throughout |
| `astropy` | BLS search (`timeseries.BoxLeastSquares`), units |
| `lightkurve` | MAST data retrieval and light curve I/O |
| `astroquery` | Catalog queries (TOI, KOI, Gaia, TIC) |
| `pydantic` | Immutable typed data models |
| `scipy` | Platt scaling optimization (Nelder-Mead) |
| `xgboost` | Tier-1 ML classifier (optional) |
| `pandas` | Training data I/O in Skills scripts |
| `matplotlib` | ROC and reliability diagram export (optional) |

Development dependencies (`pytest`, `ruff`, `mypy`) are listed in `pyproject.toml`.

---

## 5. Quick Start

### CLI

```bash
# Run the full pipeline on a TESS target
exo 150428135

# Select ML scorer with a trained model
exo 150428135 --scorer xgboost --model-path data/model.json

# Save results to JSON
exo 150428135 --output results/toi700.json
```

### Python API

```python
from exo_toolkit.fetch import fetch_lightcurve
from exo_toolkit.clean import clean_lightcurve
from exo_toolkit.search import search_lightcurve
from exo_toolkit.vet import vet_signal
from exo_toolkit.scoring import score_candidate
from exo_toolkit.pathway import classify_submission_pathway

# 1. Fetch — downloads PDCSAP photometry from MAST
fetch_result = fetch_lightcurve("TIC 150428135", mission="TESS")

# 2. Clean — sigma clipping, normalization, windowed detrending
clean_result = clean_lightcurve(fetch_result.light_curve)

# 3. Search — BLS over a period grid; returns list[CandidateSignal]
signals = search_lightcurve(clean_result.light_curve)

if signals:
    signal = signals[0]  # highest-SDE candidate

    # 4. Vet — per-transit depths, odd/even, secondary eclipse, shape
    vet_result = vet_signal(
        clean_result.light_curve,
        signal,
        stellar_radius_rsun=0.42,   # from TIC catalog
        contamination_ratio=0.01,   # from pipeline header
    )

    # 5. Score — Bayesian log-score posterior + derived scores
    posterior, scores = score_candidate(signal, vet_result.features)

    # 6. Classify — ordered gate logic → submission pathway
    pathway = classify_submission_pathway(
        signal, vet_result.features, posterior, scores
    )

    print(f"Period:   {signal.period_days:.4f} d")
    print(f"Depth:    {signal.depth_ppm:.0f} ppm")
    print(f"FPP:      {scores.false_positive_probability:.3f}")
    print(f"Pathway:  {pathway}")
```

### Training a Custom Scorer

```bash
# Download Kepler KOI table
python Skills/fetch_kepler_tce.py --output data/koi_table.csv

# Build labelled training features
python Skills/build_training_data.py \
    --koi data/koi_table.csv --output data/kepler_training.pkl

# Download TESS TOI table
python Skills/fetch_tess_toi.py --output data/tess_toi.csv

# Build TESS training features
python Skills/build_tess_training_data.py \
    --toi data/tess_toi.csv --output data/tess_training.pkl

# Merge datasets (optional)
python Skills/build_combined_training_data.py \
    --kepler data/kepler_training.pkl \
    --tess   data/tess_training.pkl \
    --output data/combined_training.pkl

# Train XGBoost with Platt calibration
python Skills/train_xgboost.py \
    --data data/combined_training.pkl --model data/model.json

# Evaluate Bayesian vs XGBoost (ROC-AUC, F1, precision, recall)
python Skills/evaluate_scorer.py \
    --data data/combined_training.pkl \
    --model data/model.json \
    --plot reports/roc.png \
    --reliability-plot reports/calibration.png
```

---

## 6. Repository Structure

```
2026-Exoplanet-Research/
├── src/
│   └── exo_toolkit/
│       ├── schemas.py           # Pydantic data models (frozen, typed)
│       ├── features.py          # RawDiagnostics; extract_features()
│       ├── hypotheses.py        # Per-hypothesis log-score functions
│       ├── scoring.py           # Softmax posterior; FPP; detection confidence
│       ├── pathway.py           # Ordered gate → 6 submission pathways
│       ├── fetch.py             # MAST retrieval via Lightkurve
│       ├── clean.py             # Sigma clipping; normalization; detrending
│       ├── search.py            # BLS search; iterative transit masking
│       ├── vet.py               # Odd/even; secondary eclipse; transit shape
│       ├── calibration.py       # Platt scaling; PAVA isotonic; Brier score
│       ├── cli.py               # `exo <TIC-ID>` Rich-formatted CLI
│       └── ml/
│           ├── xgboost_scorer.py    # XGBoost binary classifier (Tier-1)
│           └── stacking_scorer.py   # Weighted blend scorer (Tier-3)
├── tests/                       # 696 unit and integration tests
│   ├── test_schemas.py          # 33 tests
│   ├── test_features.py         # 89 tests
│   ├── test_hypotheses.py       # 28 tests
│   ├── test_scoring.py          # 25 tests
│   ├── test_pathway.py          # 35 tests
│   ├── test_fetch.py            # 40 tests (+2 live)
│   ├── test_clean.py            # 39 tests
│   ├── test_search.py           # 43 tests
│   ├── test_vet.py              # 47 tests
│   ├── test_calibration.py      # 70 tests
│   ├── test_cli.py              # 20 tests
│   ├── test_xgboost_scorer.py   # 45 tests
│   ├── test_stacking_scorer.py  # 22 tests
│   └── ...                      # Skills tests
├── Skills/                      # Reusable standalone utility scripts
│   ├── fetch_kepler_tce.py      # Download Kepler KOI cumulative table
│   ├── fetch_tess_toi.py        # Download TESS TOI table (CP/FP/EB)
│   ├── build_training_data.py   # Kepler KOI → CandidateFeatures pickle
│   ├── build_tess_training_data.py  # TESS TOI → CandidateFeatures pickle
│   ├── build_combined_training_data.py  # Merge Kepler + TESS pickles
│   ├── train_xgboost.py         # k-fold CV training with Platt calibration
│   ├── evaluate_scorer.py       # Bayesian vs XGBoost comparison (ROC, F1)
│   ├── injection_recovery.py    # Synthetic transit injection completeness maps
│   └── count_tess_labels.py     # Check CNN Tier-2 label gate (≥5,000 CP)
├── notebooks/
│   └── pipeline_demo.ipynb      # End-to-end demo on TOI-700 (TIC 150428135)
├── docs/
│   ├── SCORING_MODEL.md         # Full Bayesian scoring mathematical spec
│   ├── PIPELINE_SPEC.md         # Stage-by-stage architecture details
│   ├── ML_SCORING.md            # ML scorer modes, training pipeline, column maps
│   ├── CNN_SPEC.md              # Tier-2 CNN architecture spec (gated)
│   ├── DATA_SOURCES.md          # MAST, ExoFOP, NExSci endpoints and caching
│   ├── DECISIONS.md             # Durable design decisions with rationale
│   ├── ROADMAP.md               # Milestones and future work
│   └── PROJECT_STATUS.md        # Current implementation status
├── data/                        # Local data cache (not tracked by git)
├── pyproject.toml
└── requirements.txt
```

---

## 7. Quality Assurance

All three quality gates must pass before any commit is considered complete:

```bash
# Lint (PEP 8, import order, complexity, security)
ruff check .

# Static type checking — always use python -m mypy, not bare mypy
python -m mypy src

# Full test suite
PYTHONPATH=src python -m pytest

# All three together
ruff check . && python -m mypy src && PYTHONPATH=src python -m pytest
```

Continuous integration runs all three gates on every pull request via GitHub Actions. Live integration tests (requiring MAST network access) are excluded from CI and must be run manually:

```bash
PYTHONPATH=src python -m pytest -m integration_live
```

### Test Coverage Summary

| Module | Tests |
|---|---|
| `schemas.py` | 33 |
| `features.py` | 89 |
| `hypotheses.py` | 28 |
| `scoring.py` | 25 |
| `pathway.py` | 35 |
| `fetch.py` | 40 (+2 live) |
| `clean.py` | 39 |
| `search.py` | 43 |
| `vet.py` | 47 |
| `calibration.py` | 70 |
| `cli.py` | 20 |
| `ml/xgboost_scorer.py` | 45 |
| `ml/stacking_scorer.py` | 22 |
| Skills (injection recovery, training, evaluation, etc.) | 160 |
| **Total** | **696** |

---

## 8. Data Sources and Target Selection

| Source | URL / Access Method | Usage |
|---|---|---|
| TESS PDCSAP photometry | MAST via `lightkurve.search_lightcurve` | Primary light curves |
| Kepler/K2 photometry | MAST via `lightkurve.search_lightcurve` | Archival search |
| TESS Input Catalog (TIC) | `astroquery.mast.Catalogs` | Stellar parameters, contamination ratio |
| TESS TOI catalog | ExoFOP-TESS (`exofop.ipac.caltech.edu`) | Known-object matching; training labels |
| Kepler KOI cumulative table | NASA Exoplanet Archive DR25 (Thompson et al.) | Training labels (confirmed vs. FP) |
| CTOI catalog | NASA Exoplanet Archive | Community TOI cross-matching |
| Gaia DR3 | `astroquery.gaia` | Crowding, centroid analysis |

Rate limits and caching guidance are documented in `docs/DATA_SOURCES.md`. The TESS TOI table is downloaded as a CSV from ExoFOP-TESS and filtered to CP (confirmed planet), FP (false positive), and EB (eclipsing binary) dispositions. PC (planet candidate) entries are excluded from training labels due to label noise. The Kepler DR25 Robovetter outputs (Thompson et al.; Coughlin et al.) are used as training labels for the XGBoost classifier, restricting to `koi_disposition ∈ {CONFIRMED, FALSE POSITIVE}`.

The pipeline preferentially targets later TESS sectors, stars with TESS magnitude 10–14, and fields with low Gaia source density. This parameter space is systematically underrepresented in the published TOI catalog, maximizing the expected yield of genuinely novel candidates relative to search effort.

---

## 9. Guardrails and Scientific Integrity

This system is designed to identify **candidate signals** for follow-up investigation. The following constraints are enforced by design and must not be circumvented:

**No confirmation claims.** The pipeline never outputs "confirmed planet." All internally detected signals are labeled "candidate signal," "possible transit-like event," or "follow-up target." Confirmation status is governed exclusively by authoritative external catalogs (NASA Exoplanet Archive, TOI list, KOI list).

**Mandatory false-positive exposure.** Every `ScoredCandidate` object carries a `CandidateExplanation` that enumerates positive evidence, negative evidence, and blocking issues. Candidates may not be routed to external pathways without a populated explanation. The FPP is always displayed alongside any candidate probability estimate.

**Missing diagnostics fail conservatively.** A `None` feature score fails any gate condition it participates in rather than being treated as neutral. Specifically, `tfop_ready` requires non-`None` values for contamination, secondary eclipse, and odd/even mismatch scores; missing provenance score blocks the pathway entirely.

**Suppression of formal submission when diagnostics are absent.** If key vetting diagnostics (secondary eclipse SNR, odd/even comparison, centroid shift) are unavailable due to insufficient data coverage, the pathway classifier falls back to `planet_hunters_discussion` or `github_only_reproducibility` regardless of the posterior probability.

**ML classifiers require calibration validation.** XGBoost and ensemble scorers are not activated until Platt calibration has been fit and validated on a held-out set. Raw uncalibrated probabilities are not exposed to the pathway classifier.

**Scoring model version provenance.** Every `ScoredCandidate` output records the model name, version, commit hash, and configuration hash via `ScoringMetadata`. This ensures all published candidate scores are reproducible.

These guardrails reflect the broader ethical responsibility of automated astronomical pipelines to avoid generating premature or misleading claims that could misdirect telescope time allocation or public communication.

---

## 10. Submission Instructions

The submission pathway returned by `classify_submission_pathway()` determines the appropriate next action:

### `tfop_ready`

The candidate meets all nine TESS Follow-up Observing Program (TFOP) quality criteria. Steps:

1. Verify stellar parameters against the TIC (Stassun et al.) and 2MASS catalogs.
2. Prepare a follow-up report following the TFOP Working Group guidelines at `tess.mit.edu/followup`.
3. Submit via the ExoFOP-TESS web interface (`exofop.ipac.caltech.edu/tess`), uploading the light curve, period, epoch, and depth with uncertainties.
4. Tag the submission with the relevant TESS sector(s) and pipeline version.

### `planet_hunters_discussion`

The candidate warrants community review but does not yet satisfy formal TFOP criteria (e.g., fewer than two observed transits, moderate FPP, or missing vetting diagnostics). Steps:

1. Create a discussion thread on the [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) platform, attaching the phase-folded light curve and scoring report.
2. Note which vetting diagnostics are missing and what follow-up observations could resolve the ambiguity.

### `kepler_archive_candidate`

The candidate satisfies the Kepler/K2 quality criteria ($p_\mathrm{pc} \geq 0.65$, novelty $\geq 0.70$, FPP $\leq 0.35$). Steps:

1. Cross-check against the KOI cumulative table (Thompson et al.) to confirm novelty.
2. Submit to the NASA Exoplanet Archive Community Follow-up Program (CFOP) or post to the Kepler/K2 GitHub community repositories.

### `known_object_annotation`

The signal matches a known catalog object ($p_\mathrm{ko} \geq 0.80$). No new submission is warranted. Record the matched object ID and close the candidate.

### `github_only_reproducibility`

The candidate does not meet external submission criteria (high FPP, missing diagnostics, or fallback). Steps:

1. Open a GitHub issue in this repository with the candidate ID, TIC number, period, and scoring report.
2. Attach the phase-folded light curve image and the serialized `ScoredCandidate` JSON.
3. Label the issue `candidate-low-confidence` for future review.

---

## 11. Project Roadmap

| Status | Milestone | Details |
|:---:|---|---|
| ✅ | **Core data models** | Immutable Pydantic schemas for all pipeline types (`schemas.py`) |
| ✅ | **Feature extraction** | 35 normalized diagnostic scores; `RawDiagnostics` container (`features.py`) |
| ✅ | **Hypothesis scoring** | Per-hypothesis log-score functions for all 6 hypotheses (`hypotheses.py`) |
| ✅ | **Bayesian scorer** | Softmax posterior, FPP, detection confidence, novelty score (`scoring.py`) |
| ✅ | **Submission classifier** | Ordered gate logic → 6 submission pathways (`pathway.py`) |
| ✅ | **Data acquisition** | MAST retrieval via Lightkurve; provenance tracking (`fetch.py`) |
| ✅ | **Light curve cleaning** | Sigma clipping, normalization, windowed detrending (`clean.py`) |
| ✅ | **Transit search** | BLS periodicity search; iterative multi-planet masking (`search.py`) |
| ✅ | **Signal vetting** | Odd/even, secondary eclipse, transit shape, data-gap diagnostics (`vet.py`) |
| ✅ | **Posterior calibration** | Platt scaling, PAVA isotonic regression, Brier score (`calibration.py`) |
| ✅ | **End-to-end notebook** | Full pipeline demo on TOI-700 (TIC 150428135); candidate report |
| ✅ | **Injection-recovery** | Synthetic transit injection; completeness maps vs. period and radius |
| ✅ | **CLI entry point** | `exo <TIC-ID>` single-command pipeline invocation with Rich output |
| ✅ | **ML Tier 1 — XGBoost** | Tabular classifier on 35 `CandidateFeatures` scores |
| ✅ | **ML Tier 3 — Stacking** | Weighted blend of XGBoost + Bayesian posteriors |
| ✅ | **Platt calibration in training** | OOF predictions → `(a, b)` saved to model metadata JSON |
| ✅ | **Training data pipeline** | Kepler DR25 KOIs + TESS TOI dispositions → labelled feature pickles |
| ✅ | **Combined training dataset** | Merged Kepler + TESS pickle with stratified per-source capping |
| ✅ | **Scorer evaluation** | k-fold ROC-AUC, F1, precision, recall; ROC and calibration diagram export |
| ✅ | **CNN Tier-2 gate** | `count_tess_labels.py`; `CNN_SPEC.md` architecture document |
| 🔴 | **ML Tier 2 — 1D CNN** | Phase-folded flux classifier; gated on ≥5,000 TESS CP labels (Shallue and Vanderburg) |
| 🔴 | **Mission-specific priors** | Period-, radius-, and stellar-type-dependent priors replacing flat defaults |
| 🔴 | **Web interface / dashboard** | Interactive candidate browser with score explanations |

✅ Complete &nbsp;&nbsp; 🔴 Planned / Gated

---

## 12. License

- **Code:** Apache License 2.0 — see [`LICENSE`](LICENSE)
- **Documentation:** Creative Commons Attribution 4.0 International (CC BY 4.0)

Raw photometric data from TESS, Kepler, and K2 is provided by NASA and the MAST archive and is not relicensed by this repository.

---

## Works Cited

Astropy Collaboration, et al. "The Astropy Project: Building an Open-Science Project and Status of the 2018 Astropy Package." *The Astronomical Journal*, vol. 156, no. 3, 2018, p. 123. https://doi.org/10.3847/1538-3881/aabc4f.

Barlow, Richard E., et al. *Statistical Inference under Order Restrictions: The Theory and Application of Isotonic Regression*. Wiley, 1972.

Borucki, William J., et al. "Kepler Planet-Detection Mission: Introduction and First Results." *Science*, vol. 327, no. 5968, 2010, pp. 977–980. https://doi.org/10.1126/science.1185402.

Brier, Glenn W. "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, vol. 78, no. 1, 1950, pp. 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

Bryson, Steve, et al. "The Occurrence of Rocky Habitable-Zone Planets around Solar-Like Stars from Kepler Data." *The Astronomical Journal*, vol. 161, no. 1, 2021, p. 36. https://doi.org/10.3847/1538-3881/abc418.

Coughlin, Jeffrey L., et al. "Planetary Candidates Observed by Kepler. VII. Performance of Robovetter." *The Astrophysical Journal Supplement Series*, vol. 224, no. 1, 2016, p. 12. https://doi.org/10.3847/0067-0049/224/1/12.

Díaz, Rodrigo F., et al. "PASTIS: Bayesian Extrasolar Planet Validation — I. General Framework, Models, and Performance." *Monthly Notices of the Royal Astronomical Society*, vol. 441, no. 2, 2014, pp. 983–1004. https://doi.org/10.1093/mnras/stu601.

Fischer, Debra A., et al. "Planet Hunters: The First Two Planet Candidates Identified by the Public Using the Kepler Public Archive." *Monthly Notices of the Royal Astronomical Society*, vol. 419, no. 4, 2012, pp. 2900–2911. https://doi.org/10.1111/j.1365-2966.2011.19932.x.

Fressin, François, et al. "The False Positive Rate of Kepler and the Occurrence of Planets." *The Astrophysical Journal*, vol. 766, no. 2, 2013, p. 81. https://doi.org/10.1088/0004-637X/766/2/81.

Guerrero, Natalia M., et al. "TESS Objects of Interest Catalog from the TESS Prime Mission." *The Astrophysical Journal Supplement Series*, vol. 254, no. 2, 2021, p. 39. https://doi.org/10.3847/1538-4365/abefe1.

Hippke, Michael, and René Heller. "Optimized Transit Detection Algorithm to Search for Periodic Transits of Small Planets." *Astronomy & Astrophysics*, vol. 623, 2019, p. A39. https://doi.org/10.1051/0004-6361/201834672.

Howell, Steve B., et al. "The K2 Mission: Characterization and Early Results." *Publications of the Astronomical Society of the Pacific*, vol. 126, no. 938, 2014, pp. 398–408. https://doi.org/10.1086/676406.

Kopparapu, Ravi Kumar, et al. "Habitable Zones around Main-Sequence Stars: Dependence on Planetary Mass." *The Astrophysical Journal Letters*, vol. 787, no. 2, 2014, p. L29. https://doi.org/10.1088/2041-8205/787/2/L29.

Kovács, Géza, et al. "A Box-fitting Algorithm in the Search for Periodic Transits." *Astronomy & Astrophysics*, vol. 391, no. 1, 2002, pp. L23–L26. https://doi.org/10.1051/0004-6361:20020802.

Lightkurve Collaboration, et al. "Lightkurve: Kepler and TESS Time Series Analysis in Python." *Astrophysics Source Code Library*, 2018, record ascl:1812.013.

Morton, Timothy D. "An Efficient Automated Validation Procedure for Exoplanet Transit Candidates." *The Astrophysical Journal*, vol. 761, no. 1, 2012, p. 6. https://doi.org/10.1088/0004-637X/761/1/6.

Platt, John. "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*, edited by Alexander J. Smola et al., MIT Press, 1999, pp. 61–74.

Ricker, George R., et al. "Transiting Exoplanet Survey Satellite (TESS)." *Journal of Astronomical Telescopes, Instruments, and Systems*, vol. 1, no. 1, 2015, p. 014003. https://doi.org/10.1117/1.JATIS.1.1.014003.

Shallue, Christopher J., and Andrew Vanderburg. "Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90." *The Astronomical Journal*, vol. 155, no. 2, 2018, p. 94. https://doi.org/10.3847/1538-3881/aa9e07.

Stassun, Keivan G., et al. "The TESS Input Catalog and Candidate Target List." *The Astronomical Journal*, vol. 156, no. 3, 2018, p. 102. https://doi.org/10.3847/1538-3881/aad050.

Thompson, Susan E., et al. "Planetary Candidates Observed by Kepler. VIII. Cumulative Planet Candidate Catalog." *The Astrophysical Journal Supplement Series*, vol. 235, no. 2, 2018, p. 38. https://doi.org/10.3847/1538-4365/aab4f9.

VanderPlas, Jacob T. "Understanding the Lomb-Scargle Periodogram." *The Astrophysical Journal Supplement Series*, vol. 236, no. 1, 2018, p. 16. https://doi.org/10.3847/1538-4365/aab766.

Winn, Joshua N. "Transits and Occultations." *Exoplanets*, edited by Sara Seager, University of Arizona Press, 2010, pp. 55–77.
