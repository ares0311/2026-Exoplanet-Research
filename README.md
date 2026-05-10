# 🚀 2026 Exoplanet Research

![Status](https://img.shields.io/badge/status-active%20development-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Focus](https://img.shields.io/badge/focus-exoplanets-purple)

---

## 🌌 Overview

A **research-grade, reproducible pipeline** for detecting and evaluating exoplanet candidates from **TESS** and **Kepler** data.

### Core Flow

```
Raw Light Curves → Detection → Vetting → Bayesian Scoring → Submission Pathway
```

This project prioritizes:
- Scientific rigor
- Low false-positive rates
- Reproducibility
- High-value (potentially habitable) candidates

---

## 🧠 Key Idea

Most signals are **not planets**.

This system is built to **disprove signals first**, then elevate only the strongest candidates.

---

## 📊 Current Status

**Phase:** Foundation / Early Build

- ✅ Repo initialized
- ✅ Documentation system built
- ✅ Scoring architecture defined
- ⏳ Pipeline implementation in progress

👉 See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)

---

## 🛣 Roadmap

| Milestone | Description |
|----------|------------|
| 1 | Minimal pipeline (fetch → detect) |
| 2 | Vetting system |
| 3 | Bayesian scoring engine |
| 4 | Submission classification |
| 5 | Reporting system |
| 6 | Calibration |
| 7 | Injection–recovery |

👉 See [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## ⚙️ Architecture

```
Fetch → Clean → Search → Vet → Score → Classify
```

| Module | Purpose |
|-------|--------|
| fetch.py | Data acquisition |
| clean.py | Preprocessing |
| search.py | Transit detection |
| vet.py | False-positive rejection |
| scoring.py | Bayesian classification |
| pathway.py | Submission routing |

👉 See [`docs/PIPELINE_SPEC.md`](docs/PIPELINE_SPEC.md)

---

## 📐 Scoring Model

Bayesian framework:

```
P(H | D) ∝ P(D | H) P(H)
```

Hypotheses:
- Planet candidate
- Eclipsing binary
- Background binary
- Stellar variability
- Instrumental artifact
- Known object

Outputs:
- Posterior probabilities
- False positive probability
- Detection confidence
- Submission readiness

👉 See [`docs/SCORING_MODEL.md`](docs/SCORING_MODEL.md)

---

## 📂 Project Structure

```
src/
notebooks/
data/
docs/
tests/
```

## 🖥 Local System Profile

Local development and batch-run sizing guidance is recorded in [`docs/SYSTEM_PROFILE.md`](docs/SYSTEM_PROFILE.md).

---

## ⚠️ Important Disclaimer

This project identifies **candidate signals only**.

❌ No claims of confirmed exoplanets  
❌ No replacement for professional validation pipelines  

---

## 📜 License

- Code: Apache 2.0  
- Docs: CC-BY-4.0  

---

## 🔭 Vision

Build a system that produces:

> **Scientifically defensible, reproducible exoplanet candidates**

—not just interesting plots.
