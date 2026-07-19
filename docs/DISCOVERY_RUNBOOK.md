# DISCOVERY RUNBOOK

**Purpose**: Prevent doom loops. Every agent and every session must read this before doing anything.

**Last updated**: 2026-07-11 (the trained-model and formal-acceptance gates are
complete; `tess_live_search_v1` is now the frozen, schema-v2-ledger-wired source
for the next bounded live-search evidence run)

---

## Current Production Direction (Read This First)

The project is no longer spending its primary workflow on the run006/run008 QLP
candidate-review loop. That loop produced useful scanner/debugging evidence,
but it did not produce submission-ready candidates. The trained model, stacking
calibration, and formal production-acceptance gates are now complete. The next
active Phase 1 implementation is complete: schema-v2 candidate-ledger writes
are wired to the frozen `tess_live_search_v1` source. The next step is its
operator-coordinated live evidence run.

The authoritative active plan is now:

1. `docs/ROADMAP.md` — active master-guide phase ordering
2. `metadata/dataset_manifests/tess_live_search_v1.json` — stable live-search
   source ID and checksum
3. `data_selection/batch_manifests/tess_live_search_v1.json` — exact target,
   product, storage, and retention scope
4. `docs/PRODUCTION_READINESS.md` — closed deployment gates and version history

Use the discovery workflow below as historical/operational reference for live
scans. Do not treat it as permission to continue the old review loop unless the
user explicitly asks for discovery-scan forensics.

### Frozen TESS live-search preparation

The metadata-only command used to create v1 was:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/star_scanner.py --prepare-only --max-stars 20 --workers 6
```

It queries target and product metadata only, writes the canonical priority
queue plus immutable `tess_live_search_v1` manifests, validates their checksum,
and exits before constructing a scan log or invoking the transit pipeline. It
must fail closed if TOI, CTOI, confirmed-host, coordinate, product-URI, product-
size, storage, or manifest validation is incomplete. V1 is now committed; do
not rerun this command or use `--replace-preparation` to revise its membership.
Create new explicitly versioned output paths and IDs for a future batch. Do not
execute the 18-target raw-data scan with the legacy dynamic-selection mode; use
only `--execute-prepared-batch`, which validates this source ID and exact URI
inventory before writing any scientific ledger row.

## The Discovery Mission (Operational Reference)

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
- The pinned ASAS-SN Catalog X known-variable-star source (version 0.2.98,
  `Skills/star_scanner.py`'s `_load_asassn_variable_tic_ids()`) — a star
  already classified as a variable (eclipsing binary, pulsator, etc.) is a
  poor novel-transit-search target for the same reason a known planet host
  is: it's already characterized, just not by a planet-search pipeline.

`Skills/toi_checker.py` and `Skills/star_scanner.py` implement TOI, CTOI, confirmed-host, and known-variable exclusion. Keep these exclusions enabled before running discovery batches.

### REMINDER (added 2026-07-18, read this before any post-production data-source decision): raw Full Frame Image extraction is a separate, later, deliberately-scoped phase — not Day 1

This note exists because the user explicitly asked for a durable reminder,
anticipating forgetting it once the project reaches live production. Any
session reading this file (which the PRIMARY DIRECTIVE requires every
session to do) must surface this before treating "we've searched
everything" as true.

**What Day 1 discovery actually does today**: `select_targets()` /
`inspect_target_products()` only ever query *already-extracted* light
curves — products some pipeline (SPOC, QLP, TGLC, TASOC,
GSFC-ELEANOR-LITE, T16, TARS, etc.) has already turned into a compact
time-vs-brightness file from TESS's raw camera images. A live check in
this project (2026-07-17/18) found that essentially every priority-ranked
candidate star in the Tmag 12–14.5 band already has at least one such
light curve, almost always via QLP. This makes "scan existing light
curves" a cheap, bounded, already-built Day 1 operation — but it is
**not** an exhaustive search of TESS's data. It only sees stars someone
else's pipeline already chose to process.

**What Day 1 discovery does NOT do**: extract a *new* light curve directly
from raw Full Frame Image (FFI) pixel data for a star nobody's pipeline
has ever processed. This is a fundamentally heavier, different engineering
task — raw pixel calibration, aperture or PSF photometry, systematics/
scattered-light removal, background modeling — comparable in scope to
building a new pipeline, not to running an existing query. It is the only
way to reach the stars that are genuinely unprocessed by anyone (likely
concentrated in the faint tail below existing pipelines' magnitude cutoffs,
and in crowded/blended fields automated pipelines skip). Rough proportion
estimates given in-conversation on 2026-07-18 (by star count, and
separately by raw FFI byte-volume, both meaningfully large) were **not**
independently verified against the published QLP/TESS literature — that
verification is part of this reminder's scope, not yet done.

**Separately, a due-diligence gap**: a light curve already existing in a
public archive proves *someone's pipeline* processed the raw pixels; it
does not prove anyone screened it for a transit, and it does not fully
rule out that another individual or group already looked at a specific
candidate's light curve for exactly this purpose without publishing or
otherwise recording it anywhere this project could check. This is a
different risk than the FFI-extraction gap above and cannot be closed by
more exclusion-list plumbing alone.

**Action for a future session**: before authorizing any raw-FFI
photometric-extraction effort, treat it as its own bounded, separately
contracted production phase (matching this project's established pattern
for major new data-source work — source contract, metadata-only preflight,
bounded pilot, evidence gate, then scale) — not something folded into the
existing `select_targets()`/`run_background_scan()` path. Do not start
building it without the user's explicit go-ahead at that time.

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

## TESS Target Selection: Option B — B1-B4 Merged, B5 Historical

**Use only as historical discovery-scan provenance unless a future task
explicitly reopens live-scan forensics.**

Historical gate: `star_scanner.py` excludes TOI, CTOI, and confirmed exoplanet hosts from the NASA Exoplanet Archive, and defaults to the Tmag 12.0-14.5 novelty frontier. The first SPOC-only run completed but did not close the gate because nearly every selected TIC had no SPOC long-cadence light curve. The first QLP rerun also did not close the gate because three stale local Lightkurve cache FITS files were corrupt from interrupted downloads and the shared fetch path did not repair them before retrying. The next QLP attempt repaired cache files but still crashed because Lightkurve public download methods mutate process-global stdout under worker-thread concurrency. The stdout-safe QLP attempt completed but still did not close the gate because the shared fetch path requested SPOC-style `pdcsap_flux`; valid QLP products do not provide `PDCSAP_FLUX`. The flux-safe QLP attempt still did not close the gate because it produced third-party MAST download chatter and no durable scan log before the first completed target. Run006 completed after the progress/quiet-download and bounded-BLS fixes. Targeted run008 reproduced both run006 filtered candidates, but later v0.2.10 regeneration moved them above the prior FPP < 0.15 escalation threshold. This is useful provenance, not the active production blocker.

### Option B build plan

| Step | What to build | File |
|------|--------------|------|
| B1 | Add CTOI exclusion to `star_scanner.py::run_background_scan()` using `Skills/fetch_exofop_ctoi.py` | MERGED (PR #139) |
| B2 | Add confirmed-planet cross-check using NASA Exoplanet Archive TAP (`ps` table, `pl_tranflag=1`) | MERGED (PR #139) |
| B3 | Default `tmag_range` in `star_scanner.py` to `(12.0, 14.5)` to target faint-star novelty frontier | MERGED (PR #139) |
| B4 | Extend default `period_max` in BLS search to 500 d | MERGED (PR #139) |
| B5 | Review first 200-target QLP discovery scan and document candidate/numerical quality | HISTORICAL / NOT ACTIVE |

**Status**: B1-B4 merged to `main`. B5 run006/run008 evidence is retained for provenance, but it is no longer the highest-priority active production gate.

---

## Anti-Doom-Loop Rules

These rules exist because the same mistakes have been repeated across many sessions:

### Rule 1: No more Skills milestones
Do not write new Skills scripts unless they directly enable a running discovery scan or close a named gap in `docs/PRODUCTION_READINESS.md`. The Skills library at 415+ scripts is already overcomplete relative to the number of discovery runs executed (which is zero).

### Rule 2: Dataset-brief-first model work only
The old "no more CNN until discovery review" rule is superseded by the
2026-07-01 project reset. CNN/model work is now authorized because the user
adopted `docs/exoplanet_exomoon_dataset_handoff.md` as the route to a trained
AI model. Do not repeat C1-C19/C20-style ad hoc retraining; first verify source
contracts, schemas, manifests, leakage controls, storage estimates, and local
artifact ledger state.

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

### Rule 7: Run Report Policy — every acquisition/processing script self-reports

Every long-running acquisition or processing Skill (fetchers, batch scanners,
corpus builders — sharded or single-threaded) must, at the end of each
successful invocation (or each shard's invocation), write a small structured
completion record via `Skills/run_report.py` and auto-commit + push *only
that record* to git. This replaces the old pattern of a human pasting console
output for the agent to manually transcribe into tracking docs.

**Why**: pasted console output is lossy, easy to mis-transcribe, and clutters
chat context. A script that reports its own outcome — like a CI job posting a
status, or a PR announcing its own merge — is more reliable and scales to
many concurrently running console tabs.

**Mechanism** (`Skills/run_report.py`):
- `RunReport` — a small dataclass: script name, status (`success`/`partial`/`failed`),
  timestamps, elapsed seconds, items processed/written/failed, output paths,
  and shard index/count when sharded.
- `append_run_report(report, path)` — appends one JSON line to a ledger file
  under `artifacts/manifests/run_reports/<script>.jsonl` (or
  `<script>.shardIofN.jsonl` when sharded, so concurrent shards never contend
  for the same file — mirrors how sharded output/raw-dir paths are already scoped).
- `commit_and_push_report(path, message=...)` — stages **only** that exact
  file (never `git add .`/`-A`), commits, and pushes, retrying once via
  `git fetch` + `git rebase` on a non-fast-forward rejection. Never raises —
  a report-push failure prints a warning and exits 0; it must never crash or
  block the actual data-acquisition work that already succeeded.
- This is a narrow, intentional exception to the branch/PR/CI policy: a
  run-report push goes directly to whatever branch is checked out (normally
  `main`), bypassing the feature-branch cycle, because it only ever touches
  one small, git-tracked metadata file that the operator did not otherwise
  edit. It must never be used to push code, data, or manifest changes.

**Retrofit scope**: applies to all existing acquisition/processing Skills —
`batch_scan.py`, `star_scanner.py`, `fetch_kepler_lc_snippets.py`,
`fetch_tess_lc_snippets.py`, `fetch_tess_kepler_overlap_snippets.py`,
`fetch_tess_k2_overlap_snippets.py`, `fetch_kepler_tce.py`,
`fetch_tess_toi.py`, `fetch_exofop_ctoi.py`, `fetch_nea_koi_lc_index.py`,
`fetch_additional_tess_labels.py`, `fetch_confirmed_hosts.py`,
`fetch_jwst_targets.py`, `fetch_jwst_lc.py`, `tess_tce_fetcher.py` — not just
`process_t1_kepler_batch.py` (the first adopter). Retrofit these
incrementally; do not block using a script on it having a run report yet, but
do not consider the retrofit complete until each of the scripts above calls
`run_and_commit_report` at the end of a successful run.

**Checking progress without pasted console output**: read the ledger
directly, e.g. `.venv/bin/python Skills/run_report.py process_t1_kepler_batch`,
or ask for a fresh `--status-only` run where the script supports one (see
`process_t1_kepler_batch.py --status-only`).

### Rule 8: Parallelism-first for any recipe over ~3 minutes

Before handing the user a recipe expected to take longer than ~3 minutes,
work through whether sharding (independent units, e.g. targets/files/folds),
in-process worker concurrency (I/O-bound, e.g. `--workers` per
`docs/SYSTEM_PROFILE.md`'s 4-6 guidance for external services), or
multiprocessing (CPU-bound local work) would meaningfully cut wall-clock
time — do not default to sequential without checking. If the target script
doesn't yet support the applicable form, either add it (mirroring
`process_t1_kepler_batch.py`'s `--workers`/`--shard-index`/`--shard-count`
pattern) or say explicitly why not. Ask the operator rather than guess when
the right shard/worker count depends on their own tradeoffs (tabs they're
willing to dedicate, trust in the external service's rate limits, whether
another task is already running). See `CLAUDE.md`'s "Parallelism-First
Recipe Policy — MANDATORY" for the full rule.

**Measure before recommending more**: after any parallel/sharded run, compare
its real per-item rate against the last baseline before proposing the next
step up in shards/workers — a regressed rate or new errors is a stop signal,
not something to push past. Sub-linear scaling is itself a bug to investigate
(check for an artificial in-process bottleneck first), not just a result to
shrug off because nothing crashed — this project found exactly that twice
(`_DOWNLOAD_PRODUCTS_LOCK` in `exo_toolkit/fetch.py`, fixed in version
0.2.19). See CLAUDE.md's "Measure-then-scale cadence" for the full rule.

**Single-parent launch rule**: for reviewed 6×6 acquisition workloads, use
`.venv/bin/python Skills/run_six_shards.py` rather than asking the operator to
open six terminal tabs. Its allowlist, clean-main identity check, storage
projection, isolated logs, interruption cleanup, and serialized Run Report git
transactions are mandatory safety properties for any equivalent supervisor.
For full local validation, use `.venv/bin/python
Skills/run_quality_gates.py`; it runs six disjoint test shards × six xdist
workers plus Ruff/mypy under one parent. Use this optimized supervisor pattern
for other safely partitionable work as it is added and measured.

---

## What Has Been Built (Capability Inventory)

The pipeline can do these things today without new code:

| Capability | Command |
|------------|---------|
| Scan a single star | `exo <TIC-ID> --output out.json` |
| Scan a list of stars | `.venv/bin/python Skills/batch_scan.py targets.txt --output results.json --resume` |
| Prepare a frozen novel-TIC live-search queue | `.venv/bin/python Skills/star_scanner.py --prepare-only --max-stars 20 --workers 6` |
| Select and immediately scan novel TIC targets (legacy path; do not use for the frozen v1 queue) | `.venv/bin/python Skills/star_scanner.py --max-stars 500 --tmag-min 12 --tmag-max 15` |
| Rank candidates by quality | `.venv/bin/python Skills/rank_candidates.py results.json --top 20` |
| Check if target is already TOI | `.venv/bin/python Skills/toi_checker.py <TIC-ID>` |
| Filter by FPP/pathway | `.venv/bin/python Skills/alert_filter.py results.json --fpp-max 0.15` |
| Generate phase-fold plot | `.venv/bin/python Skills/plot_lc.py results.json --output-dir plots/` |
| Generate FP vetting report | `.venv/bin/python Skills/false_positive_vetter.py results.json --output reports/fp_vetting.md` |
| Generate review dashboard | `.venv/bin/python Skills/candidate_dashboard_export.py results.json --output reports/candidate_dashboard.html` |
| XGBoost scorer (better FP rejection) | `exo <TIC-ID> --scorer xgboost --model-path models/xgboost_koi.json` |

The XGBoost model (`models/xgboost_koi.json`) is trained and available now.

### `exo scan` interactive vs. automation-safe display modes

`exo <TIC-ID>` (Milestone 20) shows a live-updating spinner tracking the
fetch → clean → search → vet/score/classify stages plus elapsed time when
stdout is an interactive terminal. Machine-readable behavior is identical
in every mode: JSON output (`--output`), exit codes, and the candidate
summary printed after the run never depend on whether the animation ran.

- **Interactive terminal (default)**: an animated spinner shows the
  current stage and elapsed seconds.
- **Redirected output / CI / non-TTY**: animation is disabled
  automatically (`sys.stdout.isatty()` is checked once at startup) and
  plain `[<elapsed>s] <Stage> ...` lines are printed instead — safe to
  pipe, log, or diff.
- **`--no-animation`**: forces the plain-line mode even in an interactive
  terminal, for reduced-motion preference or predictable terminal
  recordings.
- **Interruption (Ctrl-C)**: the spinner (if any) stops cleanly, a single
  `Interrupted during stage: <Stage>` line is printed to stderr, and the
  process exits with code `130` — no partial JSON is written.

This is purely a terminal-presentation layer around the existing pipeline
(`run_pipeline()`'s new optional `on_stage` callback); it never changes
scoring, thresholds, or classifications.

---

## The Immediate Next Action (As of 2026-07-11)

The frozen 18-target batch is complete; do not rerun it. Audit the conservative
review queue in `artifacts/manifests/tess_live_search_v1_run_summary.json`.
The first conservative review is complete. TIC 201251996_s03 and s05 are
`likely_false_positive` because their stellar-eclipse-scale depths (58.4% and
31.7%) fail large-depth and companion-radius checks and have weak XGBoost
support. TIC 355651994_s02 (P=97.1618 d, depth=1.1546%, FPP=0.0663) is
`plausible_but_weak`: 14 available checks pass, but limb-darkening plausibility
fails and 17 diagnostics remain missing. Next obtain centroid/contamination and
per-transit odd/even evidence for that signal. Keep all language at “candidate
signal”; QLP provenance is below the `tfop_ready` gate and no external
submission is authorized. A 2026-07-12 storage preflight found the shared
Lightkurve cache at 85 GB (41 GB Kepler, 37 GB TESS, 7.3 GB HLSP, 302 MB K2),
above the 80 GB caution threshold. A metadata-only TESS-SPOC search found
exactly two target-pixel products for TIC 355651994 (sectors 1 and 28), totaling
12.25 MB. That bounded sample is not a substantial-data operation and needs no
sharding (two products, expected under three minutes); do not broaden it into a
multi-target or archive download. The first version 0.2.39 run downloaded both
files and established that neither sector covers a predicted event. Version
0.2.40 records that no-coverage result with exact time ranges rather than
raising without an artifact. The cache-only rerun completed in 8.97 seconds
and auto-committed its Run Report as `986e7bc`; do not rerun it. The
reproducibility command was:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/tpf_centroid_diagnostic.py \
  "TIC 355651994" \
  --period-days 97.16176299117727 \
  --epoch-bjd 2458417.8455470423 \
  --duration-hours 8.1 \
  --output reports/tess_live_search_v1_TIC355651994_s02_centroid.json
```

The command is single-process by design because only two independent products
exist. Its structured result is summarized in
`artifacts/manifests/tess_live_search_v1_tpf_coverage_summary.json`. Neither TPF
contains a predicted event, and the QLP review has only two independent events
versus four required for odd/even. Both checks are observationally unavailable;
additional event-covering observations would be required to resolve them.

Historical run006/run008 evidence:

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

The immediate follow-up was targeted, not another blind 200-target scan.
Run008 reran TIC 201252011 and TIC 257712351 with `Skills/star_scanner.py
--target`, `--pipeline QLP`, `--exptime long`, `--max-period-grid-points
20000`, and `logs/discovery_run_008_targeted_qlp_stitch_safe.json`. Both
targets remained `candidate_found` with FPP <= 0.15, but the vetting reports
still show many unavailable diagnostics and a `limb_darkening_plausibility_score`
failure for each best signal. The next work is candidate-specific
false-positive diagnostics, not another blind scan and not CNN training.

Do not submit or contact externally from run006/run008 without explicit human
approval. Do not use this historical loop to block the T1-1 dataset/model
training path.

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
