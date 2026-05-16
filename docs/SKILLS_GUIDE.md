# Skills Guide

All standalone utility scripts live in `Skills/`.  Each can be used as an
importable library **and** run directly from the command line.

Run any script with `--help` to see its full CLI options.

---

## Quick reference

| Script | Purpose | Key function |
|--------|---------|--------------|
| `star_scanner.py` | Rank uncharacterised TIC targets, background scan | `priority_score`, `scan_star`, `run_background_scan` |
| `batch_scan.py` | Scan a list of TIC IDs from text/CSV | `batch_scan`, `read_tic_ids` |
| `rank_candidates.py` | Sort exo JSON outputs by composite rank score | `rank_candidates`, `compute_rank_score` |
| `sector_coverage.py` | Query TESS sector availability without downloading | `get_sector_coverage` |
| `toi_checker.py` | Look up a TIC ID in the ExoFOP TOI list | `check_toi` |
| `watchlist.py` | Persistent follow-up watchlist | `Watchlist` |
| `alert_filter.py` | Filter batch results by FPP/pathway/SNR thresholds | `filter_candidates` |
| `export_candidates.py` | Export ranked candidates to CSV and Markdown | `to_csv`, `to_markdown_table` |
| `summary_report.py` | Generate Markdown summary from batch_scan output | `build_report`, `write_report` |
| `plot_lc.py` | Phase-fold PNG from candidate JSON | `plot_candidate`, `phase_fold` |
| `notebook_generator.py` | Generate Jupyter notebook for a TIC target | `generate_notebook` |
| `target_prioritizer.py` | Rank TIC IDs by TOI status + sector coverage | `prioritize_targets`, `format_recommendations` |
| `compare_candidates.py` | Merge multi-file JSON into Markdown comparison | `load_and_merge`, `build_comparison_report` |
| `candidate_timeline.py` | Track candidate score evolution across runs | `CandidateTimeline.record`, `to_markdown` |
| `fits_header_extractor.py` | Extract stellar params from TESS SPOC FITS headers | `extract_from_header`, `to_vet_kwargs` |
| `injection_recovery.py` | Measure pipeline completeness via synthetic transit injection | `run_injection_recovery` |
| `evaluate_scorer.py` | k-fold ROC-AUC, F1, reliability diagram | `evaluate_scorer` |
| `build_training_data.py` | Map Kepler KOI table → CandidateFeatures pickle | `build_training_data` |
| `build_tess_training_data.py` | Map TESS TOI table → CandidateFeatures pickle | `build_tess_training_data` |
| `build_combined_training_data.py` | Merge Kepler + TESS training pickles | `build_combined_training_data` |
| `train_xgboost.py` | Stratified k-fold XGBoost training with Platt calibration | `train_xgboost` |
| `fetch_kepler_tce.py` | Download Kepler KOI cumulative table | `fetch_kepler_tce` |
| `fetch_tess_toi.py` | Download TESS TOI table from ExoFOP | `fetch_tess_toi` |
| `count_tess_labels.py` | Check CNN Tier-2 label gate (≥5,000 CP) | (script) |

---

## Target discovery workflow

The typical discovery workflow chains several Skills:

```
star_scanner → batch_scan → alert_filter → rank_candidates → watchlist
                                                           ↘ export_candidates
                                                           ↘ summary_report
```

### 1. Discover promising targets

```bash
# Query TIC for uncharacterised K/M dwarfs; scan top 500 in priority order
python Skills/star_scanner.py --log data/scan_log.json --max-stars 500

# Resume after interruption (log tracks already-scanned)
python Skills/star_scanner.py --log data/scan_log.json --max-stars 500

# Show log summary without scanning
python Skills/star_scanner.py --summary --log data/scan_log.json
```

### 2. Scan a specific target list

```bash
# targets.txt — one TIC ID per line (or CSV with a "tic_id" column)
python Skills/batch_scan.py targets.txt \
    --output results.json \
    --resume                     # skip already-done entries on re-run
```

### 3. Check TOI status before investing pipeline time

```bash
python Skills/toi_checker.py 150428135
# Not found in ExoFOP TOI list. TIC 150428135    (unknown target)
# — or —
# TOI 700.01  |  TIC 150428135  |  CP  P = 37.4237 d
```

### 4. Filter results by quality thresholds

```bash
# Keep only tfop_ready candidates with FPP < 0.20 and ≥ 2 signals
python Skills/alert_filter.py results.json \
    --fpp-max 0.20 \
    --pathway tfop_ready \
    --min-signals 2 \
    --output filtered.json
```

### 5. Rank, export, and report

```bash
# Sort by composite rank score
python Skills/rank_candidates.py filtered.json --top 10

# Export to CSV and Markdown
python Skills/export_candidates.py filtered.json \
    --csv out/candidates.csv \
    --markdown out/candidates.md \
    --stats

# Generate Markdown summary report
python Skills/summary_report.py results.json --output reports/summary.md
```

### 6. Manage follow-up watchlist

```bash
# Add promising targets
python Skills/watchlist.py add 150428135 --note "K-dwarf, 2 signals"
python Skills/watchlist.py add 261136679

# List watchlist
python Skills/watchlist.py list

# Feed watchlist directly into batch_scan (IDs one per line)
python Skills/watchlist.py list | grep "^TIC" | awk '{print $2}' > wl.txt
python Skills/batch_scan.py wl.txt --output wl_results.json
```

---

## Visualisation

```bash
# Phase-folded PNG for all candidates in a JSON file (requires matplotlib)
python Skills/plot_lc.py results.json --output-dir plots/
```

The `phase_fold(time, flux, period, epoch)` function is also importable
directly for use in notebooks:

```python
from Skills.plot_lc import phase_fold
import numpy as np

phase, flux = phase_fold(lc.time.jd, lc.flux.value, period=5.0, epoch=2458600.0)
```

---

## Sector coverage

Before downloading light curves for a list of targets, check which sectors
are available:

```bash
python Skills/sector_coverage.py TIC 150428135
python Skills/sector_coverage.py TIC 150428135 --pipeline QLP --json
```

---

## ML training pipeline

```bash
# 1. Download labels
python Skills/fetch_kepler_tce.py --output data/koi_table.csv
python Skills/fetch_tess_toi.py   --output data/tess_toi.csv

# 2. Build feature pickles
python Skills/build_training_data.py \
    --input data/koi_table.csv --output data/kepler_features.pkl
python Skills/build_tess_training_data.py \
    --input data/tess_toi.csv --output data/tess_features.pkl

# 3. Merge
python Skills/build_combined_training_data.py \
    --kepler data/kepler_features.pkl \
    --tess   data/tess_features.pkl \
    --output data/combined_features.pkl

# 4. Train
python Skills/train_xgboost.py \
    --input data/combined_features.pkl \
    --output data/model.json

# 5. Evaluate
python Skills/evaluate_scorer.py \
    --input data/combined_features.pkl \
    --model data/model.json \
    --roc plots/roc.png
```

---

## Injection-recovery completeness

```bash
python Skills/injection_recovery.py \
    --target "TIC 150428135" \
    --n-injections 200 \
    --output data/recovery.json
```

---

## CNN Tier-2 gate check

```bash
python Skills/count_tess_labels.py
# Current CP count: 4,217  |  Gate threshold: 5,000  |  Status: BLOCKED
```

---

## Notebook generation

Generate a ready-to-run Jupyter notebook for any TIC target:

```bash
python Skills/notebook_generator.py 150428135 \
    --mission TESS \
    --stellar-radius 0.42 \
    --stellar-mass 0.40 \
    --output notebooks/TIC_150428135.ipynb
```

The notebook covers all 6 pipeline stages with prose and code cells.

---

## Target prioritization

Rank a list of TIC IDs before committing pipeline time:

```bash
# targets.txt — one TIC ID per line
python Skills/target_prioritizer.py targets.txt \
    --min-priority 0.40 \
    --skip-tois \
    --output priority.json
```

Combines TOI lookup, sector coverage, and a priority heuristic to label each
target as `scan`, `skip_toi`, or `skip_low_priority`.

---

## Multi-run candidate comparison

Compare results from different pipeline configurations or batch runs:

```bash
python Skills/compare_candidates.py \
    results_bayesian.json results_xgboost.json \
    --sort-by rank_score \
    --output reports/comparison.md
```

---

## Candidate timeline tracking

Record how a candidate's FPP and pathway evolve as new data arrives:

```python
from Skills.candidate_timeline import CandidateTimeline

tl = CandidateTimeline("data/timeline.json")
tl.record(row, note="added sector 55 data")
print(tl.to_markdown("TIC0-001"))
print(tl.summary("TIC0-001"))  # {n_runs, trend_fpp, ...}
```

---

## FITS stellar parameter extraction

Pull stellar parameters directly from a TESS SPOC FITS file for use with
`vet_signal`:

```python
from Skills.fits_header_extractor import extract_stellar_params

params = extract_stellar_params("TIC150428135_s0001_lc.fits")
vet_result = vet_signal(lc, signal, **params.to_vet_kwargs())
```

---

## Library usage pattern

All Skills follow the same convention: importable functions + `_cli()` entry
point.  Import them in scripts or notebooks without running the CLI:

```python
import sys
sys.path.insert(0, ".")  # project root

from Skills.rank_candidates import load_candidates, rank_candidates, compute_rank_score
from Skills.alert_filter import filter_candidates
from Skills.export_candidates import to_markdown_table

rows = load_candidates(["results.json"])
ranked = rank_candidates(rows, top_n=20)
filtered = filter_candidates(ranked, fpp_max=0.25, pathway="tfop_ready")
print(to_markdown_table(filtered))
```
