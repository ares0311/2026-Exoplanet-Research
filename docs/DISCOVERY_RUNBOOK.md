# DISCOVERY RUNBOOK

**Purpose**: Prevent doom loops. Every agent and every session must read this before doing anything.

**Last updated**: 2026-06-30 (Option A JWST integration and Option B TESS novelty restructure merged; PR #143 live scanner fix merged; PR #145 worker/ETA fix merged; SPOC-only B5 attempt, QLP corrupt-cache attempt, QLP stdout-race attempt, QLP wrong-flux-column attempt, and QLP no-progress/no-durable-log attempt did not close T1-0; run006 completed and now requires candidate/numerical-quality review; version 0.2.6 rejects invalid and period-boundary BLS peaks)

---

## The Mission (Read This First)

The goal is to **discover previously unknown exoplanet transit candidates** by searching photometric data that has not been thoroughly analyzed by existing automated pipelines.

The output of this project is a **ranked candidate list** that survives algorithmic false-positive rejection and is escalated to human experts for review. This is citizen science in the rigorous sense: systematic null-hypothesis rejection that produces candidate reports suitable for expert follow-up and eventual publication.

**What this is NOT:**
- A tool to re-scan known planets (TOI-700, pi Mensae, etc.)
- A tool to confirm discoveries (confirmation requires RV + HRI)
- A machine-learning research project whose primary product is an AUC score
- A Skills-writing marathon

---

## The Discovery Workflow (The Only Authorized Loop)

```
Step 1: SELECT targets from unanalyzed data feeds
         → Filter out TOI + CTOI + known exoplanet hosts
         → Prioritize: Tmag 12–15, recent sectors, long-baseline targets

Step 2: SCAN with BLS
         → Skills/star_scanner.py or Skills/batch_scan.py
         → Per-target: exo <TIC-ID> --scorer [bayesian|xgboost|ensemble]
         → Collect candidates where FPP < 0.50 and detection_confidence > 0.30

Step 3: REJECT false positives algorithmically
         → Run all diagnostic Skills on each surviving candidate
         → Target-specific analysis (centroid shift, odd/even depth, secondary eclipse)
         → If you FAIL to reject the null hypothesis → escalate

Step 4: ESCALATE surviving candidates to human review
         → Run exo <TIC-ID> --output candidate.json
         → Generate phase-fold plot (Skills/plot_lc.py)
         → Generate false-positive vetting notes (Skills/false_positive_vetter.py)
         → Generate review dashboard (Skills/candidate_dashboard_export.py)
         → Submit to Planet Hunters TESS or CTOI process only after explicit human approval
```

**If CI has not run on this session's branch, STOP — get CI green before anything else.**

---

## Where to Look: Unanalyzed Data Frontiers

These are ranked by novelty (most underexplored first):

| Priority | Target Type | Why Unanalyzed | How to Query |
|----------|-------------|----------------|--------------|
| 1 | **Tmag 12–15, any sector** | Automated pipelines deprioritize; citizen science avoids | TIC query with Tmag range in star_scanner.py |
| 2 | **Recent sectors (64–68)** | SPOC output available but citizen-science attention has moved on | `select_targets()` with recent sector filter |
| 3 | **Long-period candidates (P > 200 d)** | Most BLS searches cap at 200 d; requires multi-sector light curves | Extend `period_max` in search.py BLS call |
| 4 | **20-second cadence targets** | Expensive to process; barely touched by any survey | MAST query with `exptime=20` |
| 5 | **JWST targets** | Lightkurve does not support JWST natively; requires custom photometry | **IN SCOPE** — use `Skills/fetch_jwst_targets.py` to list programs, `Skills/fetch_jwst_lc.py` to extract flux, and `exo <obsid> --mission JWST` to run the full pipeline (MERGED PR #133, #141) |

**Novelty criterion**: A target is "novel" if its TIC ID does NOT appear in:
- ExoFOP TOI list (already flagged by TESS pipeline)
- ExoFOP CTOI list (already flagged by community)
- NASA Exoplanet Archive confirmed planets table

`Skills/toi_checker.py` and `Skills/star_scanner.py` implement TOI, CTOI, and confirmed-host exclusion. Keep these exclusions enabled before running discovery batches.

---

## What Background Automation Is For (NOT Discovery)

`src/exo_toolkit/background/` and `exo background-run-once` scan **7 static fixture targets** (3 known planets + 4 synthetics). This is a **CI validation tool**, not a discovery engine. Its purpose is to verify the pipeline runs correctly end-to-end without network access. Do not confuse this with discovery.

**Rule**: Never propose adding more fixture targets to `background/`. Never run `background-run-once` expecting to find a new planet.

---

## BLS Search Parameters for Novelty

Current defaults in `search.py` are conservative. For novel discovery, use these:

```python
period_min = 0.3    # days — catch ultra-short hot Earths
period_max = 500    # days — extended for long-period planets (was 200)
duration_min = 0.25 # hours
duration_max = 15   # hours
min_snr = 5.0       # lower threshold for faint stars (was 7–10)
```

For Tmag 12–15 specifically:
- Use multi-sector light curves (concatenated) to improve sensitivity
- Re-detrend with sigma-clip before BLS (already in `clean.py`)
- Run even-odd depth check on all candidates (already in `vet.py`)

The live scanner path uses a bounded BLS grid (`--max-period-grid-points`) so
long-baseline QLP scans remain operational on the local M4 Max without silently
launching hundreds of millions of trial periods. Do not change these thresholds
without candidate-specific evidence.

---

## The FPP Threshold for Escalation

Academic practice (Shporer & Winn 2015; Morton et al. 2016; Kunimoto & Matthews 2020):

- FPP < 0.01 = statistically validated (suitable for Planet Hunters CTOI, paper)
- FPP < 0.05 = worthy of TFOP follow-up (SG1 photometry, SG2 spectroscopy)
- FPP < 0.15 = escalate for human review (Planet Hunters TESS discussion)
- FPP ≥ 0.50 = do not escalate; discard

The `pathway.py` thresholds are set conservatively. They are correct. Do not change them without evidence from a specific candidate.

**For our pipeline output, escalate any candidate where:**
- `false_positive_probability < 0.15`
- `detection_confidence > 0.40`
- `pathway` is `tfop_ready`, `planet_hunters_discussion`, or `kepler_archive_candidate`

---

## JWST: Option A — MERGED (PR #133)

**Authorized 2026-06-26. Build A before B.**

### What JWST offers for discovery

JWST does not run autonomous surveys the way TESS does. Its time-series observations are pre-planned on specific targets. The discovery opportunities are:

1. **Serendipitous transits on background stars** — a JWST target field may contain background stars showing unnoticed transits in the same aperture data
2. **Deeper photometry of TESS candidates** — re-analyze TESS candidate hosts with JWST's superior precision to confirm or reject shallow signals
3. **Long-period candidates in JWST parallel observations** — NIRCam parallel programs observe adjacent fields opportunistically; these fields are less curated

**What is accessible via MAST (astroquery):**
- Stage 2 calibrated integrations (`_calints.fits`) — time-stamped flux per integration
- Stage 3 extracted 1D spectra (`_x1dints.fits`, NIRISS SOSS) — time-series spectral traces
- `astroquery.mast.Observations.query_criteria(obs_collection='JWST', dataproduct_type='timeseries')`

**Lightkurve does NOT support JWST natively.** The integration uses `astroquery.mast` directly and converts JWST data products to pipeline-compatible LightCurve objects.

### Option A build plan

| Step | What to build | Skill |
|------|--------------|-------|
| A1 | Query MAST for JWST time-series observations; list available programs and targets | `Skills/fetch_jwst_targets.py` |
| A2 | Download JWST calibrated integration products (`_calints.fits`); extract flux vs. time | `Skills/fetch_jwst_lc.py` |
| A3 | Convert JWST data to pipeline LightCurve format; wire into `exo` CLI with `--mission JWST` | `src/exo_toolkit/fetch.py` extension |

**Status**: A1 MERGED (PR #133). A2 MERGED (PR #133). K2 TAP ORA-00904 fix MERGED (PR #134). A3 MERGED (PR #141): `exo --mission JWST` is wired through the CLI.

### Option A constraints

- JWST data products are large (>100 MB per observation). Download only what is needed.
- JWST time units are MJD (Modified Julian Date). Convert to BTJD (BJD − 2457000) before passing to `search.py`.
- JWST calibration pipeline version varies by program. Use Stage 3 products preferentially.
- Do NOT attempt to run BLS on JWST spectral time series (x1dints) directly — extract white-light curve first by summing across wavelength.

---

## TESS Target Selection: Option B — B1-B4 Merged, B5 Review Needed

**Use for first real discovery-scan review.**

Current gate: `star_scanner.py` excludes TOI, CTOI, and confirmed exoplanet hosts from the NASA Exoplanet Archive, and defaults to the Tmag 12.0-14.5 novelty frontier. The first SPOC-only run completed but did not close the gate because nearly every selected TIC had no SPOC long-cadence light curve. The first QLP rerun also did not close the gate because three stale local Lightkurve cache FITS files were corrupt from interrupted downloads and the shared fetch path did not repair them before retrying. The next QLP attempt repaired cache files but still crashed because Lightkurve public download methods mutate process-global stdout under worker-thread concurrency. The stdout-safe QLP attempt completed but still did not close the gate because the shared fetch path requested SPOC-style `pdcsap_flux`; valid QLP products do not provide `PDCSAP_FLUX`. The flux-safe QLP attempt still did not close the gate because it produced third-party MAST download chatter and no durable scan log before the first completed target. Run006 completed after the progress/quiet-download and bounded-BLS fixes; T1-0 is now blocked on candidate/numerical-quality review, not another blind scan. Version 0.2.6 adds a fail-closed BLS guard for invalid peaks and period-grid boundary peaks, directly addressing the run006 negative-duration error and boundary-period artifact class.

### Option B build plan

| Step | What to build | File |
|------|--------------|------|
| B1 | Add CTOI exclusion to `star_scanner.py::run_background_scan()` using `Skills/fetch_exofop_ctoi.py` | MERGED (PR #139) |
| B2 | Add confirmed-planet cross-check using NASA Exoplanet Archive TAP (`ps` table, `pl_tranflag=1`) | MERGED (PR #139) |
| B3 | Default `tmag_range` in `star_scanner.py` to `(12.0, 14.5)` to target faint-star novelty frontier | MERGED (PR #139) |
| B4 | Extend default `period_max` in BLS search to 500 d | MERGED (PR #139) |
| B5 | Review first 200-target QLP discovery scan and document candidate/numerical quality | REVIEW NEEDED |

**Status**: B1-B4 merged to `main`. B5 run006 is complete locally and review-blocked; this is the highest-priority active production gate.

---

## Anti-Doom-Loop Rules

These rules exist because the same mistakes have been repeated across many sessions:

### Rule 1: No more Skills milestones
Do not write new Skills scripts unless they directly enable a running discovery scan or close a named gap in `docs/PRODUCTION_READINESS.md`. The Skills library at 415+ scripts is already overcomplete relative to the number of discovery runs executed (which is zero).

### Rule 2: No more CNN training until discovery runs produce candidates
CNN training consumed 6 weeks (C1–C19). The CNN improves FP rejection. FP rejection is only useful if there are candidates to reject. Run at least 1,000 TIC targets through the discovery loop before resuming CNN training. If zero candidates emerge, CNN does not matter.

### Rule 3: Background automation is not discovery
`exo background-run-once` runs on fixtures. It does not search new sky. Do not mention it in a discovery session.

### Rule 4: Do not ask the user questions you can look up
If the question is "what FPP threshold does academia use?", look it up. If the question is "which TESS sectors are least searched?", look it up. Only escalate to the user when the decision requires their personal judgment (budget, risk tolerance, which expert to contact).

### Rule 5: Every session starts from synced `main`
Every recipe given to the user must begin by switching to `main` and fast-forwarding from `origin/main`. Never pull `origin/main` into a feature branch and never give a command from a branch that has not been merged to main.

### Rule 6: Training data cleanup
Rejected CNN training artifacts consume disk space and create confusion. After any training attempt is formally rejected (not just failed — formally documented as REJECTED in AGENTS.md):

```bash
# Remove rejected checkpoint (keep only the reference hash in AGENTS.md)
rm -rf checkpoints/cnn_tess_c*/best.pt       # rejected checkpoints
rm -rf checkpoints/cnn_tess_finetuned/       # rejected fine-tune

# Remove intermediate training data splits that can be regenerated
rm -rf data/tess_cnn_splits/                 # can regenerate from snippets
rm -rf data/tess_combined_cnn_splits/        # can regenerate
rm -rf data/tess_c20_cnn_splits/             # can regenerate

# Keep these (source data, cannot easily regenerate):
# data/tess_snippets_v2.jsonl               — keep
# data/kepler_snippets.jsonl                — keep
# data/tess_combined_snippets.jsonl         — keep
# data/tess_k2_overlap_snippets.jsonl       — keep when built
# checkpoints/cnn_kepler_pretrain/best.pt   — keep (Kepler pretrain, val AUC 0.9186)
```

**When to run this**: After each formal REJECTION is documented. Do not run speculatively. Confirm with the user before deleting anything larger than 1 GB.

---

## What Has Been Built (Capability Inventory)

The pipeline can do these things today without new code:

| Capability | Command |
|------------|---------|
| Scan a single star | `exo <TIC-ID> --output out.json` |
| Scan a list of stars | `.venv/bin/python Skills/batch_scan.py targets.txt --output results.json --resume` |
| Select novel TIC targets | `.venv/bin/python Skills/star_scanner.py --max-stars 500 --tmag-min 12 --tmag-max 15` |
| Rank candidates by quality | `.venv/bin/python Skills/rank_candidates.py results.json --top 20` |
| Check if target is already TOI | `.venv/bin/python Skills/toi_checker.py <TIC-ID>` |
| Filter by FPP/pathway | `.venv/bin/python Skills/alert_filter.py results.json --fpp-max 0.15` |
| Generate phase-fold plot | `.venv/bin/python Skills/plot_lc.py results.json --output-dir plots/` |
| Generate FP vetting report | `.venv/bin/python Skills/false_positive_vetter.py results.json --output reports/fp_vetting.md` |
| Generate review dashboard | `.venv/bin/python Skills/candidate_dashboard_export.py results.json --output reports/candidate_dashboard.html` |
| XGBoost scorer (better FP rejection) | `exo <TIC-ID> --scorer xgboost --model-path models/xgboost_koi.json` |

The XGBoost model (`models/xgboost_koi.json`) is trained and available now.

---

## The Immediate Next Action (As of 2026-06-29)

**Review run006 candidate evidence** (higher priority than CNN training):

The first real QLP discovery scan completed locally as
`logs/discovery_run_006_qlp_progress_safe.json`. It produced 200 entries:
192 `candidate_found`, 6 `scanned_clear`, 1 `no_data`, 1 `error`, and
0 active targets. Filtering with `--fpp-max 0.15` produced
`logs/discovery_filtered_006_qlp_progress_safe.json` with two rows:

| TIC | Period (d) | FPP | Pathway |
|---|---:|---:|---|
| TIC 201252011 | 227.39056281978395 | 0.1160636155807766 | `planet_hunters_discussion` |
| TIC 257712351 | 142.95415231096942 | 0.12672985673564718 | `planet_hunters_discussion` |

Before any external action, review the filtered candidates and the full scan
log. The run is useful evidence, but not submission-ready: 192/200 targets were
flagged as candidates and 81 detections landed at the 0.5 d or 500 d period
boundaries, so the next work is candidate/numerical-quality review and
false-positive diagnostics.

Version 0.2.6 rejects invalid BLS peaks and peaks pinned to the BLS period-grid
boundary. Treat any run006 candidate review as pre-0.2.6 evidence; any future
evidence rerun must start from synced `main` at 0.2.6 or newer.

The immediate follow-up should be targeted, not another blind 200-target scan:
rerun TIC 201252011 and TIC 257712351 with `Skills/star_scanner.py --target`,
`--pipeline QLP`, `--exptime long`, `--max-period-grid-points 20000`, and a
fresh log path such as `logs/discovery_run_007_targeted_qlp_v026.json`. The
targeted path records the active target before live work begins and prints
flushed start/completion lines, so the operator can tell whether the process is
alive.

Do not build the C20 CNN corpus or train C20 until this discovery run has been
reviewed. Do not submit or contact externally without explicit human approval.

---

## Submission Pathway Reference

When a candidate survives the discovery loop:

| Pathway label | What it means | Next human action |
|--------------|---------------|-------------------|
| `tfop_ready` | Meets all TFOP WG SG1 conditions | Submit CTOI to ExoFOP; request SG1 ground photometry |
| `planet_hunters_discussion` | Promising but missing one TFOP condition | Post to Planet Hunters TESS forum for community vetting |
| `kepler_archive_candidate` | Strong Kepler/K2 candidate | File a KOI/K2OI via NASA Exoplanet Archive |
| `github_only_reproducibility` | Low confidence; needs more data | Document in repo only; do not submit externally |
| `known_object_annotation` | Already known | Add to watchlist; skip external submission |

**ExoFOP CTOI submission URL**: https://exofop.ipac.caltech.edu/tess/ctoi.php  
**Planet Hunters TESS**: https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess

No external submission without explicit human approval. This is a hard constraint enforced by the `background/` module and by CLAUDE.md.
