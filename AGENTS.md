# AGENTS.md — Instructions for AI Coding Agents

This file contains binding rules for AI coding agents working in this repository.

---

## PRIMARY DIRECTIVE — READ THIS BEFORE ANYTHING ELSE

**The only authorized work is work that advances this project to live production.**

Every session must begin by reading:
1. `AGENTS.md` (this file)
2. `docs/PRODUCTION_READINESS.md`
3. `docs/DISCOVERY_RUNBOOK.md`
4. `docs/exoplanet_exomoon_dataset_handoff.md`
5. `docs/exoplanet_detection_research_brief.md` (skim satellite table + AI methods)
6. `docs/astrometrics_coding_agents_master_guide.md`
7. `docs/astrometrics_data_selection_policy.md`
8. `docs/astrometrics_external_and_cloud_storage_policy.md`

Before proposing or executing any task you must:
1. Name the highest-priority unresolved Tier 1 gap from `docs/PRODUCTION_READINESS.md`.
2. State explicitly how the proposed work closes or directly unblocks that gap.
3. If the proposed work does not close or unblock a named gap in `docs/PRODUCTION_READINESS.md`, **do not do it**.

### Prohibited work

- Adding Skills, modules, schemas, or scaffolding that do not directly close a named Tier 1 or Tier 2 gap.
- Repeating work already listed under "What Is Complete" in `docs/PRODUCTION_READINESS.md`.
- Writing "the next N utility scripts" when those scripts do not unblock a named gap.
- Treating "Apply All System Directives" as permission to add more code — it means read the gap list and work the highest-priority gap only.
- Running `exo background-run-once` expecting to discover new planets — background automation scans **7 static fixture targets** (3 known planets + 4 synthetics) and is a CI validation tool, not a discovery engine. See `docs/DISCOVERY_RUNBOOK.md §Background Automation`.
- Continuing the run006/run008 candidate-review loop as the primary production path. Those scans are now historical evidence; the active production path is the dataset/model-training plan in `docs/exoplanet_exomoon_dataset_handoff.md`.
- Proposing ad hoc CNN retraining against the old rejected corpora without first satisfying the dataset/source-contract requirements in `docs/exoplanet_exomoon_dataset_handoff.md`.

### When the user says "Apply All System Directives"

1. Read `AGENTS.md` and `docs/PRODUCTION_READINESS.md`.
2. State the current Tier 1 and Tier 2 gaps in priority order.
3. For planning: propose tasks in priority order where **every task closes or unblocks a named gap**. Stop when gap-closing tasks run out — do not pad the list with non-gap work. Tasks may be agent-led (code) or human-led (data collection, API keys, expert review, network access) — both are valid plan items. Label each task clearly: **[AGENT]** or **[HUMAN]**.
4. For each task, identify external dependencies (API keys, network access, GPU, human reviewer) and surface them as explicit questions before the DO phase.
5. Do not propose or execute work that does not close a named gap.

### Two-phase workflow: PLAN then DO

**PLAN phase** ("plan the next N tasks"):
- List all gap-closing tasks in priority order, labeled **[AGENT]** or **[HUMAN]**.
- For every **[HUMAN]** task, provide exact step-by-step instructions so the human can act independently.
- Ask all questions about external dependencies upfront.
- Do not execute anything.

**Between PLAN and DO — resolve all [HUMAN] tasks first:**
- The human works through every **[HUMAN]** task using the instructions from the plan.
- All **[HUMAN]** blockers must be cleared before the DO phase begins.
- If a **[HUMAN]** task needs interactive help, work through it with the human until it is resolved.

**DO phase** ("DO the next N tasks"):
- By the time DO begins, all **[HUMAN]** blockers are already cleared.
- Execute only **[AGENT]** tasks.
- The DO phase should never contain a **[HUMAN]** blocker — if one appears, the PLAN phase was incomplete.

### Outside blockers are not code problems

If the highest-priority Tier 1 gap is blocked by a human action (data collection, network access, API key, expert review), state the gap, name the blocker, and **immediately provide a complete step-by-step recipe** assuming the user has zero background knowledge of the specific task. Do not ask "do you want the commands?" — give them.

### Human-blocker recipe format

When the user must take an action to unblock a gap:
1. Give exact commands to copy-paste, in order, with no ambiguity
2. Explain what each command does in one plain-English sentence
3. State exactly what output to paste back so you can continue
4. Do not stop at "here's how to get started" — give the complete recipe through to the handoff point

---

## HANDOFF STATE — 2026-07-01 (READ THIS FIRST)

**Mission realignment (2026-07-01):** The run006/run008 QLP candidate-review loop is no longer the active production path. It produced useful evidence, but the v0.2.10 regenerated candidate packets moved the two filtered candidates above the prior FPP < 0.15 escalation threshold and did not produce submission-ready evidence. Stop spending the main workflow on that loop unless a future task explicitly asks for forensic review.

**Active production path:** Wholly adopt `docs/exoplanet_exomoon_dataset_handoff.md` as the source contract for getting a trained model over the hump. The highest-priority active Tier 1 gap is now **T1-1: Production Tier 2 CNN / trained model checkpoint**. Work must focus on verified public data access, immutable source snapshots, training manifests, leakage-safe splits, bounded storage, and a production-grade trained exoplanet classifier/ranker. Exomoon work is Track B residual/anomaly ranking only, not a supervised confirmed-exomoon classifier.

**Research brief wired (2026-06-27):** `docs/exoplanet_detection_research_brief.md` is now required reading. Key takeaways: TESS > Kepler/K2 > JWST in discovery priority order; CNN architecture baseline is Shallue & Vanderburg (2018); PLATO launches end-2026 (pipeline should handle long-baseline photometry); GP for correlated noise; citizen submissions require full transit + false-positive diagnostic table.

**Dataset handoff brief wired (2026-07-01):** `docs/exoplanet_exomoon_dataset_handoff.md` is now required reading and the authoritative data/ML strategy. Core rules: no guessed schemas/URLs/columns, no synthetic training data for this phase, no Kaggle mirrors when primary NASA/MAST sources are available, no unverified pretrained weights, no bulk archive downloads without storage estimates and human approval, and preserve enough metadata to redownload raw files after cleanup.

**Astrometrics policy docs wired (2026-07-08):** The master guide, data-selection policy, and external/cloud-storage policy in `docs/` are now required reading. Apply them to T1-1 work: separate training/validation/calibration/frozen-eval/live-search roles, record data decisions in `data_selection/data_selection_decision_log.md`, keep raw public archives as bounded re-downloadable cache rather than permanent Dropbox-synced truth, preserve manifests/ledgers/model cards, and use the 4TB external SSD as the normal large local workspace when available.

**Historical discovery work:** JWST integration A1-A3, K2 TAP fixes, and TESS target restructuring B1-B4 are merged. Run006/run008 evidence remains useful as live-pipeline evidence but is no longer the main production blocker.

| Item | State |
|---|---|
| Option A1 — `Skills/fetch_jwst_targets.py` | **MERGED** (PR #133) |
| Option A2 — `Skills/fetch_jwst_lc.py` | **MERGED** (PR #133) |
| K2 TAP ORA-00904 fix | **MERGED** (PR #134, 2026-06-27) |
| Option A3 — `--mission JWST` wired into `exo` CLI | **MERGED** (PR #141, 2026-06-27) |
| Option B1–B4 — TESS target restructuring | **MERGED** (PR #139, 2026-06-27) |
| Live scanner startup/target-selection hardening | **MERGED** (PR #143, 2026-06-28) |
| Option B5 — first 200-target discovery scan | **HISTORICAL / NOT ACTIVE** — run006/run008 produced useful evidence, but v0.2.10 candidate regeneration moved the two filtered candidates above FPP 0.15; do not continue as the main loop |
| K2 overlap corpus (`data/tess_k2_overlap_snippets.jsonl`) | **COMPLETE** — 2,086 snippets (2026-06-27) |
| Dataset/model-training handoff brief | **ACTIVE CONTRACT** — `docs/exoplanet_exomoon_dataset_handoff.md` |

**T1-1 promoted on 2026-07-09:** the master-corpus Kepler CNN checkpoint passed the held-out gates (AUC 0.9572, calibrated F1 0.8347), received explicit human approval, and is promoted as the frozen `benchmark_cnn_v1` artifact under `models/cnn/benchmark_cnn_v1/`. Do not restart old CNN experiments unless a named validation gate fails. The promotion PR is merged and CI-clean; the active Tier 1 gap is now T1-2 stacking calibration, in progress (manifest + catalog-only scoring + snippet fetcher built 2026-07-10; live snippet fetch is the next human action — see item 9 below).

**Binding T1-1 roadmap after the new policy docs:**
1. **[AGENT COMPLETE] Promotion tooling compatibility:** `Skills/promote_cnn_checkpoint.py` accepts the current `method: temperature` calibration JSON from `Skills/evaluate_cnn_checkpoint.py`, preserves legacy Platt support, and prints the intentional `git add -f` checkpoint staging step required by the Git-Add-Safe Artifact Policy.
2. **[AGENT COMPLETE] Model evidence package:** `models/benchmark_cnn_v1/MODEL_CARD.md` and `models/benchmark_cnn_v1/REPRODUCIBILITY_MANIFEST.json` record the selected local checkpoint, calibration/config/metrics hashes, source/split/config links, runtime/Python/PyTorch/MPS assumptions, limitations, and exact future `git add -f` scope.
3. **[AGENT COMPLETE] Data-role and storage records:** `data_selection/data_role_registry.yaml`, `data_selection/data_selection_decision_log.md`, `docs/LOCAL_ARTIFACT_LEDGER.md`, and `artifacts/manifests/local_artifacts.json` make training, validation, calibration, frozen-eval, storage, retention, and re-download policy visible from GitHub.
4. **[AGENT COMPLETE] Benchmark designation:** the promoted architecture/data/preprocessing bundle is designated as the frozen `benchmark_cnn_v1` measuring stick; after promotion, do not casually tune this checkpoint family.
5. **[HUMAN COMPLETE] Promotion approval:** human approved promotion of checkpoint SHA `f29e6891c255289fa1e2eddad1fb6ca131c063cf11c24b8113e0e29d049441c5` as `benchmark_cnn_v1` on 2026-07-09.
6. **[AGENT COMPLETE] Artifact promotion:** selected checkpoint/calibration/config/metrics/manifest artifacts are copied into `models/cnn/benchmark_cnn_v1/`, `models/registry.json` is updated, and ignored checkpoint artifacts are intentionally staged with `git add -f`.
7. **[AGENT COMPLETE] T1-2 held-out calibration manifest and catalog-only scoring (2026-07-10):** `Skills/build_t1_2_k2_calibration_manifest.py` builds a leakage-safe, disjoint (EPIC-namespace) held-out set from `k2pandc` — 596 rows (356 CONFIRMED / 240 FALSE POSITIVE), committed at `metadata/t1_2_k2_calibration_manifest.jsonl`. `Skills/score_t1_2_k2_calibration.py` scores every row with the existing trained `models/xgboost_koi.json` and the Bayesian log-score model (catalog-only, no light curve needed), writing `metadata/t1_2_k2_calibration_partial_predictions.jsonl`. Two real unit bugs were found and fixed while building this (see `docs/PRODUCTION_READINESS.md` T1-2 and `data_selection/data_selection_decision_log.md`'s 2026-07-10 entry): `pl_tranmid` is already full BJD (not BKJD, which had corrupted `Skills/fetch_tess_k2_overlap_snippets.py`'s 2,086 already-fetched TESS-domain K2-overlap snippets — that local file needs re-fetching before any future use, no promoted model affected), and `pl_trandur` is empirically in hours despite the archive's own schema metadata claiming days.
8. **[AGENT COMPLETE] Native K2 snippet fetcher built (2026-07-10):** `Skills/fetch_t1_2_k2_calibration_snippets.py` fetches native K2 light curves (not TESS) via the production `exo_toolkit.fetch.fetch_lightcurve()` path, with resume, terminal-failure logging, bounded `--workers`, ETA output, and Run Report Policy integration. 8 offline tests pass. Not yet run live.
9. **[HUMAN NEXT] Run the live K2 snippet fetch:** `caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py --manifest metadata/t1_2_k2_calibration_manifest.jsonl --output data/t1_2_k2_calibration_snippets.jsonl --workers 4` (~600 targets, much smaller than the 6,515-target Kepler batch).
10. **[AGENT NEXT] After the live fetch:** merge `cnn_prob` into the partial predictions using `CnnScorer.from_checkpoint("models/cnn/benchmark_cnn_v1/best.pt")` against the fetched snippets, run `Skills/calibrate_stacking_weights.py` on the completed predictions to produce `models/stacking_weights.json`, then update `StackingScorer`'s default weights per that tool's printed recipe through the normal branch/PR/CI cycle.

### What was done in the previous sessions (2026-06-21 – 2026-06-26)

- **Temperature scaling live** — PR #125 merged. Platt replaced by temperature scaling in evaluator.
- **C17 REJECTED** — joint Kepler+TESS training (9,633 examples) achieved only val AUC 0.7859, worse than C13–C15 (0.81–0.84). Root cause: domain mismatch. Do not retry joint training.
- **C18 REJECTED** — `freeze_conv_epochs=10` on TESS combined splits. Best candidate so far: test AUC 0.8439, test F1 0.7979 (raw). Failed gates: raw AUC 0.8439 < 0.85 (short by 0.006); T=1.61 worsened already-excellent test calibration (ECE 0.0301→0.0667). See runbook Step 7e.
- **C19 REJECTED** — `freeze_conv_epochs=20` (doubled from C18). SHA-256 `65f3721fac577807f35e4edaeaa9cc0cd0f50959441344487f7c77f35a570436`. Test AUC 0.8420 < C18's 0.8439 — regressed. T=1.88 worsened ECE further (0.0377→0.0760). Root cause: LR scheduler does not fire during the frozen phase (val_auc improving), so conv unfreezes at same LR=1e-4 as C18; longer frozen phase over-adapted the FC head. See runbook Step 7f.
- **freeze_conv strategy exhausted** — C18 (freeze 10) was better than C19 (freeze 20). No further freeze_conv variant is expected to break through the 0.8439 ceiling on current corpus.
- **Strategic decision (human)** — more data. The only unexploited TESS-domain labeled source is the K2 EPIC overlap corpus (K2 planets/FPs with TESS re-observations). See runbook Step 7g.
- **ECE-skip gate fix (2026-06-22)** — `evaluate_cnn_checkpoint.py` now skips temperature scaling when raw test ECE < 0.05. Root cause of C11–C19 calibration doom loop confirmed: val overconfident from early-stopping → T>1 fitted → applied to already-calibrated test → structurally worse ECE. With the fix, C20 gate is: raw AUC ≥ 0.85 AND raw F1 ≥ 0.80 (cal==raw when ECE < 0.05). Three new tests verify the skip path.
- **K2 fetcher written (2026-06-22)** — `Skills/fetch_tess_k2_overlap_snippets.py` committed.
- **K2 TAP schema discovery fix (2026-06-26, PR #131)** — first fetch attempt crashed with HTTP 400 because k2pandc uses `disposition`/`pl_orbper`/`pl_tranmid` not the guessed names. Added `_discover_k2_columns()` to query `tap_schema.columns` at startup.
- **K2 TAP query encoding fix (2026-06-26, PR #132)** — second HTTP 400: `'FALSE+POSITIVE'` in the SQL IN clause was not decoded as a space inside the SQL string literal by the TAP server. Fixed by: (1) dropping the disposition filter from SQL entirely — fetch all rows with valid period/epoch, filter by disposition locally; (2) using `urllib.parse.quote()` for proper percent-encoding of the query string.
- **K2 TAP `epic_id` column bug (2026-06-26, on-branch fix)** — third HTTP 400: `ORA-00904: 'EPIC_ID': invalid identifier`. Root cause: `epic_id` appears in `tap_schema.columns` as an ADQL view alias but the underlying Oracle column is `k2c_objid`. **NEVER use `epic_id` in a k2pandc query.** Fixed by adding `k2c_objid` as the primary candidate (before `epic_id`) in `_K2_COL_CANDIDATES`, with `epic_candname` as fallback (parsed from "EPIC 211311380.01" → 211311380). Also switched from `format=json` to `format=csv` + `urlencode` (spaces as `+`, which NASA TAP prefers). Schema discovery now logs ALL available columns to stderr for future debugging.
- **C20 config committed (2026-06-22)** — `configs/cnn_tess_c20.json` (identical to C18, freeze_conv_epochs=10, checkpoint_dir=checkpoints/cnn_tess_c20).
- **Project version bumped to 0.2.0** — `pyproject.toml` updated; "citizen-science" keyword removed; status updated to "4 - Beta".
- **Project version bumped to 0.2.1** — patch release for the production-blocking QLP scanner stdout-race fix; this is a package version change only, not a `.venv` rename.
- **Project version bumped to 0.2.2** — patch release for the production-blocking QLP flux-column fix; QLP products do not provide `PDCSAP_FLUX`, so the fetcher now uses QLP-native corrected flux columns before falling back to SAP.
- **Project version bumped to 0.2.3** — patch release for the production-blocking QLP scanner observability fix; scan logs are created immediately, active targets are checkpointed separately from completed entries, and Astroquery MAST download banners are disabled under the Lightkurve per-product path.
- **Project version bumped to 0.2.4** — patch release for discovery triage guardrails; `rank_candidates.py` and `alert_filter.py` now fail closed with operator guidance instead of raw tracebacks when a live scan log is missing or incomplete.
- **Project version bumped to 0.2.5** — patch release for bounded live discovery search; BLS period grids are capped by default so long-baseline QLP light curves do not generate hundreds of millions of trial periods, and the pipeline now passes `vet_signal(light_curve, signal)` in the documented order.
- **Project version bumped to 0.2.6** — patch release for live discovery numerical guardrails; BLS peaks with invalid values or period-grid boundary periods fail closed instead of becoming candidate signals.
- **Project version bumped to 0.2.7** — patch release for targeted run006 follow-up observability; `Skills/star_scanner.py --target` now records an active target before live work starts and prints flushed start/completion progress for explicit TIC candidate review.
- **Project version bumped to 0.2.8** — patch release for targeted run006/run008 review hardening; QLP stitching no longer uses Lightkurve's implicit pre-clean normalizer, and `exo --output` now serializes computed vetting `features` so `Skills/false_positive_vetter.py` can evaluate actual diagnostics.
- **Project version bumped to 0.2.9** — patch release for candidate-review diagnostics; `exo --output` now serializes raw vetting `diagnostics`, fetch provenance, and missing-feature names, and `Skills/false_positive_vetter.py` explains why review-blocking diagnostics are unavailable.
- **Project version bumped to 0.2.10** — patch release for candidate-review regeneration reliability; `fetch_lightcurve()` now retries transient MAST/Lightkurve connection disconnects with bounded polite backoff.
- **Project version bumped to 0.2.11 (2026-07-02)** — supplementary pipeline-correctness patch, not part of the active T1-1 dataset-handoff path. `run_pipeline()` called `vet_signal(light_curve, signal)` with zero catalog keyword arguments, so `stellar_radius_rsun`, `stellar_mass_msun`, `contamination_ratio`, and every score derived from them were always `None`/solar-default — including `limb_darkening_plausibility_score`, which silently used 5778 K for every target because `vet_signal()` did not even accept a `stellar_teff_k` parameter. New `fetch_tic_stellar_params()` in `fetch.py` does one TIC catalog lookup per TESS scan (injectable `stellar_params_fn` for tests; fails open to all-`None` on any error; not called for Kepler/K2/JWST). This is a correctness fix to the already-shipped Bayesian/XGBoost/ensemble scorers, not a reopening of T1-0 as the active path; run006/run008 remain historical per the 2026-07-01 reset below.
- **Project version bumped to 0.2.12 (2026-07-02)** — source-contract hardening patch for the active T1-1 path. The shipped verifier already fixed the live NASA Exoplanet Archive `TAP_SCHEMA.columns.table_name` case-sensitivity bug with `UPPER(table_name) = UPPER(...)`; this patch updates `docs/exoplanet_exomoon_dataset_handoff.md` and regression tests so future agents do not copy the stale exact-match schema snippets that caused the `CUMULATIVE` smoke-test failure.
- **Project version bumped to 0.2.13 (2026-07-02)** — leakage-safe manifest planning patch for the active T1-1 path. `Skills/build_t1_training_manifest.py` verifies the KOI schema, queries confirmed/false-positive Kepler KOI rows, assigns all KOIs from the same KIC to one deterministic split, writes committed manifest/summary metadata, and records the raw-FITS cleanup policy before any bulk download is requested.
- **Project version bumped to 0.2.14 (2026-07-02)** — bounded Kepler-first processing batch patch for the active T1-1 path. `Skills/process_t1_kepler_batch.py` consumes `metadata/t1_1_kepler_training_manifest.jsonl`, fetches each unique KIC's Kepler light curve once (reusing the proven phase-fold/normalisation math from `Skills/fetch_kepler_lc_snippets.py`), phase-folds every KOI row sharing that target, and writes snippets to `data/processed/t1_1_kepler_snippets/kepler_snippets.jsonl`. Progress/resume state lives in SQLite at `logs/t1_1_kepler_processing.sqlite3`; a target is only marked `done` after its snippets are flushed, so an interrupted run never leaves partial/duplicate output. Raw FITS downloads are scoped to `data/raw/t1_1_kepler_lc` and the directory is wiped after every target (success or failure) — local raw storage never exceeds roughly one target's data at a time, satisfying the storage cap by construction. `--max-targets` defaults to 25 so the first invocation is a small bounded batch, not the full 6,515-target corpus. 27 new tests, all offline/injectable. Not yet run against live Kepler data — this agent's sandbox has no Lightkurve/MAST access.
- **Project version bumped to 0.2.15 (2026-07-02)** — the first live 25-target Mac run (PR #168) measured ~49s/target, meaning a 250-target run takes ~3.4 hours; the per-target progress line now prints an ETA (`elapsed=Xs ETA=YmZZs`), not just elapsed time, so a multi-hour batch never looks hung. 8 new tests for the ETA formatter and its wiring into `run_batch`.
- **Project version bumped to 0.2.16 (2026-07-02)** — `process_t1_kepler_batch.py` gains bounded `--workers` concurrency (default 1, matching `fetch_kepler_lc_snippets.py`'s existing convention; `docs/SYSTEM_PROFILE.md` recommends 4-6 for this external-service workload), using the same `ThreadPoolExecutor` pattern already proven in that sibling script. Building this surfaced a real concurrency-correctness bug: the raw-FITS fetcher wiped one shared scratch directory after every target, which would have let one worker's cleanup delete a different worker's still-downloading files under `workers > 1`. Fixed by giving each target its own `raw_dir/target_<id>` subdirectory. 6 new tests, including a real-thread test asserting concurrent download directories never collide.
- **Project version bumped to 0.2.17 (2026-07-03)** — crash fix for the live `--workers 6` run (corrected 2026-07-04: originally mislabeled `--workers 4` in this entry; the human confirms every live run in this history used `--workers 6`, not 4), which failed with `ValueError: I/O operation on closed file`. Root cause: `make_default_lc_fetcher()` called Lightkurve's public `search.download_all()`, which is decorated with `suppress_stdout` and mutates process-global `sys.stdout` — the exact same failure class already diagnosed and fixed once before in this project for `star_scanner.py` (see the run003 note below). Fix: extended `fetch.py`'s already-proven-safe `_download_collection_with_cache_repair()` / `_download_one_with_cache_repair()` with an optional `download_dir` parameter (backward compatible, default `None`), and rewired `make_default_lc_fetcher()` to call that path instead of `download_all()`. That helper never mutates `sys.stdout`; it monkey-patches `Observations.download_products(verbose=False)` under a module-level lock instead. Rewrote the affected tests to mock at the `_download_collection_with_cache_repair` boundary and added an explicit regression test, `test_never_calls_download_all`, that fails loudly if any future edit reintroduces the unsafe call. The interrupted 250-target/`--workers 6` run left a few targets in `active` (not `done`) state in SQLite — resuming with the same command is safe and will not duplicate output.
- **Project version bumped to 0.2.18 (2026-07-03)** — two additions to `process_t1_kepler_batch.py`, both user-requested to remove the need for many console tabs to compete for the same target list:
  - **Process-level sharding**: new `--shard-index`/`--shard-count` flags let multiple console tabs run concurrently against disjoint target sets (partitioned by `target_id % shard_count`), for throughput beyond one process's `--workers` ceiling. All concurrently running tabs share one `--db-path` (now WAL-mode SQLite with a 30s busy timeout, since it's now a genuinely concurrent multi-process writer) but each shard gets its own auto-suffixed output file and its own raw-download scratch subdirectory, so shards can never collide on a file the way the 0.2.16→0.2.17 crash did. New `--status-only` flag prints the combined done/active/written/failed summary across all shards without starting a batch. 17 new tests, including a real-thread test proving two concurrent shards never touch each other's raw-download directory.
  - **Run Report Policy** (new `Skills/run_report.py`, 18 tests): every run now writes a small structured completion record and auto-commits + pushes *only that file* to git (`--no-git-report` to opt out), replacing the pattern of pasting console output for the agent to manually transcribe into tracking docs. See `docs/DISCOVERY_RUNBOOK.md` Rule 7 and `CLAUDE.md`'s "Run Report Policy — MANDATORY" section for the full policy and the retrofit list of other acquisition scripts still to be wired up (`batch_scan.py`, `star_scanner.py`, `fetch_kepler_lc_snippets.py`, and others — not yet done).
- **Project version bumped to 0.2.19 (2026-07-03)** — fixes the intra-process download-serialization bug that explained why `--workers 6` only ever gave ~9-14% over sequential (see the 2-shard live-test writeup below for the full root cause and fix). `exo_toolkit/fetch.py`'s `_download_one_quietly()` no longer wraps the entire download call in a lock; `Observations.download_products` is now wrapped once, idempotently, and never restored, instead of being monkey-patched-and-restored under `_DOWNLOAD_PRODUCTS_LOCK` (removed) on every call. 4 new tests in `tests/test_fetch.py`, including a real 2-thread test proving concurrent calls now overlap instead of serializing.
- **Project version bumped to 0.2.20 (2026-07-03)** — prep for the planned 4-concurrent-shard live test (agreed with the human, one deliberate step up from the only live data point at 2 shards, per the Measure-then-scale cadence rather than jumping straight to the 6-7 tabs available). New `test_four_concurrent_shards_never_collide` in `tests/test_process_t1_kepler_batch.py` runs 4 real concurrent threads through `run_batch()` with `shard_count=4`, verifying disjoint raw-download directories, exactly-once target coverage across all 4 shard output files, and correct global `done` state in the shared SQLite store -- offline validation of the 4-way partition logic before the live run. No functional code changes; `shard_index`/`shard_count` were already N-way generalized (never hardcoded to 2).
- **Project version bumped to 0.2.21 (2026-07-03)** — CI caught a real bug in the 0.2.20 prep test before it could hit the live 4-shard run: `T1KeplerProcessingStore._connect()` runs `PRAGMA journal_mode=WAL;` on every connection, but SQLite only permits one connection to perform that mode transition at a time. Four threads (or four real shard processes) constructing the store for the same fresh `db_path` at nearly the same instant race for that exclusivity and lose with `sqlite3.OperationalError: database is locked` -- immediately, not smoothed over by the connection's `timeout=30.0` parameter, since the transition itself needs momentary exclusivity rather than ordinary write-lock waiting. Fixed with `_ensure_wal_mode()`: a small retry loop (10 attempts, 0.2s apart) tolerating the concurrent-startup race, falling back gracefully to the default journal mode if it never succeeds within that window (WAL is a concurrency nicety here, not a correctness requirement). 2 new tests: 8 real threads concurrently constructing stores for one fresh db with no errors, and confirming the fallback never raises when every attempt fails. Would have been a real risk for the planned 4-tab live test if the tabs were started close enough together in time -- this is exactly why the offline 4-shard test was worth writing before the live run, not just documentation.
- **Project version bumped to 0.2.22 (2026-07-04)** — two changes:
  - **Manifest-progress percent-complete** in `process_t1_kepler_batch.py`: the startup banner, every run's final "Done in Xs" line, `format_batch_summary()`, and `--status-only` (when `--manifest` exists) now print `n_done/n_total (XX.X%)` against the full manifest, computed via a fresh `store.done_target_ids()` query at completion so it reflects other concurrently-running shards' progress too, not just this process's own contribution. User-requested so a running percent is visible on every invocation without a separate `--status-only` call. New `BatchSummary` fields `n_targets_done_total`/`percent_done` (defaulted for backward compatibility). 8 new tests.
  - **Real 4-shard and 6-shard live test results, self-reported via the Run Report Policy** (found on `git pull`, not pasted console output): 4-shard run (`--workers 4` each) processed 1,000 targets (250/shard) in ~2,149s combined wall-clock (max across shards) = **~2.15s/target** -- a dramatic improvement over the pre-0.2.19 2-shard baseline (17.1s/target), confirming the 0.2.19 download-lock fix unlocked real multi-worker concurrency within each shard, not just across shards. The very next 6-shard run processed 1,500 targets (250/shard, one shard had 1 row failure) in ~3,161s = **~2.11s/target** -- essentially flat versus the 4-shard rate. Per the Measure-then-scale cadence, this flat (not improving) result at 6 shards is the stop signal: going further than ~4-6 shards is unlikely to help and risks MAST throttling without new evidence it's tolerated. Recommend treating 4-6 shards as the practical ceiling for this workload absent a new reason to test higher. Cumulative done, computed from these reports plus the historical 877-before-2-shard baseline: 877 + 500 (2-shard) + 1,000 (4-shard) + 1,500 (6-shard) = **3,877/6,515 (59.5%)** -- self-verify precisely with the next run's own printed percent now that this is built.
- **Project version bumped to 0.2.23 (2026-07-04)** — the 0.2.22 percent-complete feature only reached the console output, not the committed run report itself: reading `artifacts/manifests/run_reports/process_t1_kepler_batch*.jsonl` later (as happened when computing the 59.5% figure above) still required manually summing multiple shard files against a remembered baseline. `Skills/run_report.py`'s `RunReport` gains generic optional fields `items_done_total`/`items_total`/`percent_done` (`None` when a script has no "total universe" concept, so this is opt-in per script, not mandatory), `format_run_report()` renders them when present, and `process_t1_kepler_batch.py`'s CLI now populates them from `BatchSummary`. Reading the single latest run report for any shard now shows the true up-to-date global percent directly, no summing required. 4 new tests (the PR description said 5; corrected here).
- **T1-1 Kepler-first manifest processing reached 100% (2026-07-04)**, computed from every self-reported run report on disk (877 historical baseline + 5,638 summed `items_processed` across all reports = 6,515, verified exactly against the manifest file's own unique target count, not recalled). Shards 0-3 of the last 6-way split each explicitly confirmed with a `0 processed` round; shards 4-5 hadn't shown that confirming round yet as of this check, so treat as "at or essentially at 100%" pending one more `--status-only` or run to fully confirm the last few targets, not yet a hard-certified 100%.
- **Project version bumped to 0.2.24 (2026-07-04)** — found while confirming the manifest completion: `Skills/build_cnn_training_data.py` (the tool that will build train/val/test splits before CNN training) did its own random group-based split, completely ignoring the leakage-safe `split` field that `Skills/build_t1_training_manifest.py` already assigned per KIC group and that `process_t1_kepler_batch.py`'s snippet records carry through. Running it as-is on the finished Kepler-first corpus would have silently discarded that leakage-safety guarantee. Fixed: `split_examples()` now respects a predefined `split` field when every example carries one (verifying group consistency -- a group with two disagreeing predefined splits raises `ValueError` rather than being silently resolved, since that would itself indicate a leakage bug), and falls back to the existing random group-split for corpora without one (full backward compatibility with every historical ad hoc corpus this tool has processed). New `split_source` field (`"predefined"` or `"random_grouped"`) in the written manifest and summary for transparency. 7 new tests; also corrects a doc gap where the 0.2.23 PR forgot to update the test-baseline count.
- **Project version bumped to 0.2.25 (2026-07-04)** — the human's first real live run of the 0.2.24 fix crashed immediately with `ValueError: group 'tic:0' has inconsistent predefined splits ('train' vs 'test')` on the actual 7,442-row combined T1-1 corpus. Root cause, verified from source before fixing (not guessed): `build_cnn_training_data.py`'s `_group_id()`/`_tic_id()` only ever recognized `kepid`/`tic_id`/`original_tic_id`/`group_id` -- fields from the *old* ad hoc corpora. `process_t1_kepler_batch.py`'s `_snippet_record()` writes `target_id` and `group_key` instead, a schema this tool had never been updated to recognize. None of the recognized keys matched, so every single row fell through to the same fallback group `"tic:0"`, and the new predefined-split consistency check (correctly) flagged the resulting fake cross-split "conflict." Notably, this same field-mismatch bug pre-dated the 0.2.24 fix and would have silently dumped the entire corpus into one randomly-shuffled group under the old logic -- a much worse silent failure that the new check exposed instead of hiding. Fixed: `_group_id()` now recognizes `group_key` (used verbatim -- already a complete key like `"kepler:kic:10797460"`) ahead of the older fallback chain; `_tic_id()` falls back to `target_id` when `tic_id`/`original_tic_id` are both absent (existing `tic_id=0` placeholder behavior for old ad hoc Kepler rows is unchanged, since `dict.get`'s default only applies when a key is missing, not merely falsy); `_fallback_example_id()` no longer mislabels a Kepler `target_id` row as `"TIC..."`. 4 new regression tests reproduce the exact live schema and crash.
- **T1-1 Kepler CNN checkpoint PASSED production gates (2026-07-04)** — after the 0.2.25 fix, the human built `data/t1_1_kepler_cnn_splits/` (validator PASS, `split_source: predefined`, train/val/test = 5,148/1,141/1,153) and trained `checkpoints/cnn_t1_1_kepler/best.pt` from scratch (not fine-tuned) using the already-proven `configs/cnn_kepler_pretrain.json` architecture. Best epoch 22, val AUC 0.9148. `Skills/evaluate_cnn_checkpoint.py` result: raw test AUC 0.9252, calibrated F1 0.8281, temperature scaling T=1.0 (no calibration degradation, since none was needed). This is the first checkpoint in the entire C1-C19 + T1-1 history to clear both the 0.85 AUC and 0.80 F1 gates. Promotion to `models/` requires explicit human approval per the Git-Add-Safe Artifact Policy and has not yet been granted.
- **Project version bumped to 0.2.26 (2026-07-04)** — with the gate already cleared, the human asked whether the Kepler corpus could still be grown further. Investigation found `Q1_Q17_DR24_TCE` (NASA Exoplanet Archive TAP) has real, populated `av_training_set` labels (`PC`=3,600, `AFP`=9,596, `NTP`=2,541, `UNK`=4,630 excluded) across 20,367 rows / 12,669 unique KICs -- confirmed live, not assumed; the same column exists but is entirely empty in the newer `Q1_Q17_DR25_TCE` delivery, a real dead end ruled out before committing engineering effort to DR24. A live overlap check against the committed manifest found 7,091 genuinely new KIC targets not already used -- more than the entire current 6,515-target manifest. New `Skills/build_t1_dr24_tce_expansion_manifest.py` reuses `build_t1_training_manifest.py`'s leakage-safe grouping/split helpers and `verify_dataset_sources.py`'s schema-verification pattern, excludes any KIC already present in `metadata/t1_1_kepler_training_manifest.jsonl`, and maps `PC`->1 / `AFP`->0 / `NTP`->0. `Skills/process_t1_kepler_batch.py` needed no changes at all -- its `--manifest` flag already accepts an arbitrary manifest path and `_snippet_record()` reads fields generically. 7 new offline tests (exclusion of already-known targets, UNK exclusion, multi-TCE-per-star split consistency, period/epoch/duration sanity filtering, schema-failure fast-stop). Not yet run live -- next `[HUMAN]` action is to run the new builder for real (catalog-only query, no FITS download) and confirm `flag: OK`.
- **Project version bumped to 0.2.27 (2026-07-09)** — human approved promotion of the master Kepler CNN checkpoint as `benchmark_cnn_v1`. `Skills/promote_cnn_checkpoint.py` now accepts an explicit `--model-id`, preserving the approved benchmark identity instead of deriving only a hash-based ID. Selected artifacts are promoted under `models/cnn/benchmark_cnn_v1/`, registered in `models/registry.json`, and force-added intentionally because model checkpoint paths are ignored by default. T1-1 is complete after this promotion PR is merged and CI-clean; the next Tier 1 gap is T1-2 stacking calibration.
- **DR24 expansion manifest run live and fully processed in one pass (2026-07-04)** — the manifest builder ran live: `flag: OK`, 8,213 rows across 4,760 new target groups (the drop from the earlier 7,091-new-KIC overlap estimate is fully explained by the `UNK`-label exclusion and period/epoch/duration sanity bounds, confirmed via the console's own `[15737/15737]` processed-row count matching PC+AFP+NTP exactly -- not a bug). The human then ran 6 shards x 6 workers (36 total concurrent connections -- the human's real standing cadence; this project's own earlier changelog entries had mislabeled every historical live run as `--workers 4`, corrected in the same PR) against the full manifest and it completed entirely in a single invocation: 4,760/4,760 processed (100%), 8,207 snippets, 6 rows failed, ~1.21s/target combined.
- **Master-corpus Kepler CNN checkpoint trained, evaluated, and promoted (2026-07-04 to 2026-07-09)** — the new DR24 snippets were combined with the existing corpus (7,442 + 8,207 = 15,649 rows) and CNN splits rebuilt (`data/t1_1_kepler_master_cnn_splits/`, validator PASS, `split_source: predefined`, train/val/test = 10,800/2,393/2,456, ~21.8% positive overall vs. the original corpus's ~37%). `checkpoints/cnn_t1_1_kepler_master/best.pt` trained from scratch on this larger, more class-imbalanced corpus using the same proven config and **strictly superseded `checkpoints/cnn_t1_1_kepler/` on every metric** despite the harder class balance: test AUC 0.9572 (vs 0.9252), calibrated F1 0.8347 (vs 0.8281), Brier 0.0580 (vs 0.1052, nearly halved), ECE 0.0142 (vs 0.0441, nearly a third). `Flag: PASS`. Human approved promotion on 2026-07-09; selected artifacts are registered as `benchmark_cnn_v1` under `models/cnn/benchmark_cnn_v1/`.
- **Kepler/TESS label-source completeness investigation (2026-07-05)** — prompted by the human directly challenging whether all labeled Kepler/TESS data had actually been found, and by the human dropping `docs/seti_labeled_hit_data_research.md` (a research note whose "Comprehensive Protocol" section is a reusable TAP-schema-based discovery + VizieR/literature-audit methodology for exhaustively searching mission label sources; its SETI/Breakthrough Listen section is a separate, unrelated calibration track with its own "no usable per-hit labeled table found" conclusion). Running that protocol's broader table-discovery query (not just `%tce%`) surfaced the full `_KOI` table family (`Q1_Q6/Q8/Q12/Q16_KOI`, `Q1_Q17_DR24/DR25_KOI`, `Q1_Q17_DR25_SUP_KOI`) and `k2pandc`, none of which added new usable Kepler data (see `CLAUDE.md`'s new Label-Source Discovery Protocol section for the full current-status summary: Kepler is close to exhausted via NASA Exoplanet Archive TAP; TESS has three open, unresolved threads -- the TEV TCE catalog's data API still not found, Planet Hunters TESS/NotPlaNET real labels confirmed not publicly downloadable, and the very recent T16 Planet Hunt paper's data availability unverified due to arXiv/IOPscience blocking automated fetches). No new usable data was obtained from this investigation itself; its value was verifying completeness and surfacing concrete next leads, not a data gain.
- **Project version bumped to 0.2.27 (2026-07-05)** — the human asked to promote `checkpoints/cnn_t1_1_kepler_master/` but first harden the code against cross-mission misuse, since a live grep confirmed zero mission-aware gating existed anywhere in the CNN scoring path (`run_pipeline()` applied whatever `--cnn-checkpoint` was given regardless of `--mission`). `CnnTrainingConfig` gains an optional `training_mission` field (backward compatible: old `config.json` files without it load as `None`); `Skills/train_cnn.py` gains a `--mission TESS|Kepler|K2|JWST` flag that stamps it. `CnnScorer` exposes a `.training_mission` property (read from the checkpoint's sibling `config.json`, with an explicit constructor/`from_checkpoint()` override available for retrofitting pre-existing checkpoints). `run_pipeline()`/`exo scan` now refuse by default to score a mission with a CNN checkpoint whose declared (or undeclared/`None`) training mission doesn't match, raising a clear `ValueError` naming the mismatch; `allow_cross_mission_cnn=True` / `--allow-cross-mission-cnn` is the explicit, documented escape hatch for deliberate out-of-domain testing. Unknown-mission checkpoints (including every one trained before this field existed) are blocked by the same default, on the reasoning that "we don't know" should fail closed, not open. 4 existing `test_cli.py` CNN-scorer mocks needed `training_mission = "TESS"` added (a bare `MagicMock()` attribute access is truthy and not `None`, so it would otherwise have incorrectly tripped the new guard) -- fixed as an intentional test update reflecting the new contract, not a workaround. 18 new tests across `test_cnn_training_config.py`, `test_cnn_scorer.py`, `test_train_cnn.py`, and `test_cli.py`; full project suite (2,544 tests) re-run clean with zero regressions.

### Where things stand

| Item | State |
|---|---|
| TESS v2 snippets (`data/tess_snippets_v2.jsonl`) | **COMPLETE** — 2,619 snippets on user's Mac |
| Kepler snippets (`data/kepler_snippets.jsonl`) | **LOCAL VALIDATED** — 6,837 finite snippets on user's Mac |
| Kepler splits (`data/kepler_cnn_splits/`) | **LOCAL VALIDATED** — train 4,741 / val 1,060 / test 1,036 |
| Kepler pretraining checkpoint (`checkpoints/cnn_kepler_pretrain/best.pt`) | **LOCAL PRETRAINED** — SHA `c782d7af...`; best val AUC 0.9186 |
| Combined TESS splits (`data/tess_combined_cnn_splits/`) | **VALIDATED** — train 4,892 / val 1,049 / test 1,033 |
| Joint CNN splits (`data/joint_cnn_splits/`) | **LOCAL VALIDATED** — built 2026-06-21; retain for reproducibility; do not retrain from |
| C17 checkpoint (`checkpoints/cnn_tess_c17/`) | **REJECTED** — val AUC 0.7859; domain mismatch; do not retrain |
| C18 checkpoint (`checkpoints/cnn_tess_c18/`) | **REJECTED** — test AUC 0.8439, F1 0.7979 (raw); T=1.61 worsened ECE; SHA `d33c15f4...`; best candidate of 19 |
| C19 checkpoint (`checkpoints/cnn_tess_c19/`) | **REJECTED** — test AUC 0.8420, F1 0.7951 (raw); T=1.88 worsened ECE; SHA `65f3721f...`; regressed from C18 |
| ECE-skip gate fix | **LIVE** — `evaluate_cnn_checkpoint.py` updated 2026-06-22; 53/53 tests pass |
| `Skills/fetch_tess_k2_overlap_snippets.py` | **FIXED (2026-06-26)** — schema discovery + percent-encoded query; drop IN clause, filter locally |
| `configs/cnn_tess_c20.json` | **COMMITTED** — 2026-06-22; identical to C18, freeze_conv_epochs=10 |
| K2 overlap corpus (`data/tess_k2_overlap_snippets.jsonl`) | **COMPLETE** — 2,086 snippets (2026-06-27; wrote=2086, skipped=174, terminal_failures=135, elapsed=2531s) |

### First action for the incoming agent

**Do not continue the run006/run008 candidate-review loop as the main workflow.**
The 2026-07-01 project reset wholly adopts
`docs/exoplanet_exomoon_dataset_handoff.md` as the active production plan for
getting a trained AI model. Read that brief, then close T1-1 by implementing
source-contract-first data/model hardening: verified schemas and URLs,
immutable source snapshots, training manifests, leakage-safe splits, bounded
storage, and a production-gated trained classifier/ranker.

**Concrete next step (2026-07-02 — read before doing anything else):**
`Skills/verify_dataset_sources.py` (version 0.2.12) implements the dataset
handoff doc's "Minimum access smoke test" exactly: it queries
`TAP_SCHEMA.columns` before trusting any column name, fetches sample rows from
the `cumulative` and `toi` TAP tables plus the ExoFOP public TOI CSV, and
confirms Lightkurve can find a Kepler and a TESS light curve for one real
target from those rows.

**First live run found and fixed a real bug (2026-07-02, root cause confirmed
live, not guessed):** the `cumulative` schema check failed with zero matched
columns and no error, while `toi` passed. Root cause, confirmed by having the
human run raw `curl` against `TAP_SCHEMA.tables` directly (bypassing this
tool's code entirely): the live archive registers the table as `"CUMULATIVE"`
(upper case). `TAP_SCHEMA.columns.table_name` is an exact-match string
column, not a resolved SQL identifier, so `where table_name = 'cumulative'`
silently matched nothing. Fixed by querying
`where UPPER(table_name) = UPPER('...')` — a general case-insensitivity fix,
not a hardcoded `'CUMULATIVE'` special case, since a future table could be
registered in either case. The human re-ran the fixed query live via `curl`
and got a real 2,366-byte column list back (`kepid`, `kepoi_name`, etc.). This
project has hit this exact class of TAP case-sensitivity bug before (see the
K2 TAP `epic_id`/`k2c_objid` note above) — treat any new TAP table
introspection the same way: verify live, never assume the catalog's stored
case matches the doc's.

**Full source smoke PASS (2026-07-02):** the exact verifier ran end-to-end from
the project `.venv` and returned `Overall: PASS`. Measured outputs: 5 KOI rows,
5 TOI rows, 8,064 ExoFOP public TOI CSV rows, sample KIC `10797460` with 17
Kepler light-curve search results, and sample TIC `182943944` with 21 TESS
light-curve search results. The source-access blocker is cleared. Do not ask
the human to rerun this before the next T1-1 step unless a later source change
invalidates the contract.

**Storage/source snapshot PASS (2026-07-02):**
`Skills/plan_t1_training_batch.py` writes committed source snapshots and sample
download metadata without downloading FITS files. Live run with `sample_size=5`
verified `cumulative` rows=9,564, `toi` rows=7,931, `pscomppars` rows=6,298,
KOI label rows=7,454 across 6,515 unique KICs (2,740 confirmed / 4,714 false
positive), TOI ephemeris rows=7,824 across 7,535 TICs, and ExoFOP public TOI
CSV rows=8,064. MAST search metadata estimated all-KOI Kepler long-cadence raw
FITS at 47,099,384,640 bytes (43.86 GiB) and all-TOI TESS raw FITS at
44,994,438,720 bytes (41.90 GiB), combined 92,093,823,360 bytes under the
100 GiB working cap.

**Leakage-safe Kepler manifest PASS (2026-07-02):**
`Skills/build_t1_training_manifest.py` writes
`metadata/t1_1_kepler_training_manifest.jsonl` and
`metadata/t1_1_kepler_manifest_summary.json` from verified NASA Exoplanet
Archive KOI rows without downloading FITS files. Live run with seed 42 produced
7,454 KOI rows across 6,515 target groups, with all rows for a KIC assigned to
one split: train 5,155 rows (label0=3,268 / label1=1,887), val 1,143
(721 / 422), test 1,156 (725 / 431). Summary flag `OK`, leakage errors `[]`.
Cleanup policy is explicit: raw FITS under `data/raw/t1_1_kepler_lc` are
deleted only after processed snippets validate, the manifest is OK, the
top-level SQLite processing log has no incomplete active targets, and the
operator confirms failed raw FITS are not needed for debugging.

**Bounded Kepler-first processing batch built (2026-07-02, version 0.2.14):**
`Skills/process_t1_kepler_batch.py` consumes
`metadata/t1_1_kepler_training_manifest.jsonl`, fetches each unique KIC's
Kepler light curve once, phase-folds every KOI row sharing that target,
writes snippets to `data/processed/t1_1_kepler_snippets/kepler_snippets.jsonl`,
and tracks progress/resume in SQLite at `logs/t1_1_kepler_processing.sqlite3`.
Raw FITS downloads are scoped to `data/raw/t1_1_kepler_lc` and the directory
is wiped after every target (success or failure), so raw storage never
accumulates beyond roughly one target's data. The first live Mac run
processed 25 targets in 1,216s, wrote 26 snippets, failed 0 rows, left
SQLite summary `done|25|26|0`, and verified `data/raw/t1_1_kepler_lc` was
empty (`0B`) after completion. Version 0.2.15 added an ETA to the per-target
progress line (`elapsed=Xs ETA=YmZZs`), since a multi-hour batch must never
look hung.

**Second live run PASS (2026-07-02):** `--max-targets 250` completed in
7,647s (2h7m): 250 targets processed, 268 snippets written, 0 failed rows,
SQLite consistent throughout, raw scratch dir empty after completion. 277 of
6,515 targets are now done, 6,238 remain. This confirms the tool holds up at
10x the original smoke-test scale, not just on the initial 25-target run.

**Bounded worker concurrency added (2026-07-02, version 0.2.16):**
`process_t1_kepler_batch.py` now accepts `--workers` (default 1, sequential,
matching `fetch_kepler_lc_snippets.py`'s existing convention) using the same
`ThreadPoolExecutor` pattern already proven in that sibling script.
`docs/SYSTEM_PROFILE.md` recommends 4-6 workers for this kind of
external-service/live-catalog workload on this machine. Adding this exposed
and fixed a real concurrency-correctness bug before it could reach
production: the raw-FITS fetcher previously wiped one *shared* scratch
directory after every target, which would have let one worker's cleanup
delete a different worker's still-downloading files. Each target now gets
its own `raw_dir/target_<id>` subdirectory, deleted independently after that
target finishes. 41 offline/injectable tests pass (6 new, including one that
runs three fetches on real threads and asserts their download directories
never collide).

**Correction (2026-07-04):** every `--workers 4` figure in this section (and
in `docs/PRODUCTION_READINESS.md`) was a documentation error. The human
reports having run `--workers 6` on every one of these live tests, not 4 --
this project's own written record simply mislabeled the actual command used.
The elapsed times, snippet counts, and per-target rates below are real
(taken from actual timestamps/counts in the run reports) and are unaffected;
only the worker-count label attached to them was wrong. Total-concurrency
figures derived from the (wrong) worker=4 assumption are corrected below.

**The first live `--workers 6` attempt crashed (2026-07-02/03) — fixed in
version 0.2.17.** The crash was `ValueError: I/O operation on closed file`,
caused by `make_default_lc_fetcher()` calling Lightkurve's public
`search.download_all()`, which mutates process-global `sys.stdout` via a
`suppress_stdout` decorator and is therefore unsafe under concurrent worker
threads — the exact same failure class already documented for
`star_scanner.py` below (the run003 QLP attempt). The fix reuses `fetch.py`'s
already-proven-safe `_download_collection_with_cache_repair()` download path
(module-lock-guarded, does not touch `sys.stdout`) instead of `download_all()`.
A new regression test (`test_never_calls_download_all`) asserts this stays
fixed. The interrupted run's SQLite progress is safe to resume — targets that
were mid-flight are left `active`, not `done`, and are retried automatically.

**First post-fix `--workers 6` run PASS (2026-07-03, version 0.2.17):** the
human re-ran `--max-targets 250 --workers 6` on the fixed code. No crash.
Startup banner and per-target lines confirmed both the worker count
(`workers=6`) and per-completion `elapsed=...s ETA=...` are printed exactly
as designed — verified from the pasted console output, not asserted from
memory. Result: 250 targets processed, 288 snippets written, 0 rows failed,
6,962s elapsed (1h56m). Cumulative: 280 done before this run + 250 this run =
530/6,515 done, 5,985 remaining. Notably, 6 workers only beat the prior
sequential 250-target run (7,647s) by ~9%, not ~6x — confirms
`docs/SYSTEM_PROFILE.md`'s point that this workload is MAST-network-bound,
not CPU-bound, so `--workers` helps modestly rather than linearly. `--workers
6` is now live-validated and safe to keep using at this scale.

**Second consecutive clean `--workers 6` run (2026-07-03, version 0.2.17):**
same recipe, same result shape: 250 targets processed, 278 snippets written,
0 rows failed, 6,583s elapsed (1h50m). Cumulative: 530 done before this run +
250 this run = 780/6,515 done, 5,735 remaining. No crash, no throttling
observed across two consecutive concurrent runs.

**Sharding added (2026-07-03, version 0.2.18):** for throughput beyond one
process's `--workers` ceiling, `--shard-index`/`--shard-count` let multiple
console tabs run concurrently against disjoint target sets. All tabs share
one `--db-path`; each shard writes its own auto-suffixed output file and its
own raw-download subdirectory, so tabs never collide.

**First live 2-shard test PASS (2026-07-03) — self-reported via the new Run
Report Policy, not pasted console output:** the human ran `--shard-index 0
--shard-count 2 --workers 6` and `--shard-index 1 --shard-count 2 --workers
6` concurrently in two tabs. Both shards' run reports auto-committed and
pushed themselves
(`artifacts/manifests/run_reports/process_t1_kepler_batch.shard{0,1}of2.jsonl`):
shard 0 processed 250 targets (273 snippets, 0 failed) in 8,541s; shard 1
processed 250 targets (277 snippets, 0 failed) in 8,573s. Since both ran
concurrently, effective combined wall-clock is ~8,573s (2h22m53s) for 500
targets total = ~17.1s/target combined, versus the best single-tab
`--workers 6` rate of 26.3s/target — a real ~35% combined throughput gain
from sharding, though short of the 2x a fully independent doubling would
give (~13.2s/target). Cumulative: 780 done before this test + 500 this test =
**1,280/6,515 done, 5,235 remaining**.

**Root cause of the sub-linear scaling, found and fixed same-day (version
0.2.19):** `exo_toolkit/fetch.py`'s `_download_one_quietly()` wrapped the
*entire* download call (`search._download_one(**kwargs)`) inside
`_DOWNLOAD_PRODUCTS_LOCK`, not just the monkey-patch attribute swap needed to
force Astroquery's `verbose=False`. That fully serialized every download
within one process regardless of `--workers` count -- explaining both why
`--workers 6` alone only ever gave ~9-14% over sequential, and why 2 shards
(2 separate processes, 2 separate lock instances) only got partial rather
than full 2x scaling: each shard's own internal 6 workers were still queuing
on that shard's own lock. Verified from the source (line numbers, not
guessed): the lock wraps `search._download_one(**kwargs)` directly. Fixed by
replacing the lock-guarded monkeypatch-then-restore pattern with an
idempotent, non-restoring wrap (`_ensure_download_products_quiet()`):
`Observations.download_products` is wrapped once (tagged via a marker
attribute so repeat calls are a no-op) to always force `verbose=False`, and
is *never* restored to the noisy original, since there is no scenario in
this codebase where that's wanted mid-process. A benign race (two threads
both wrapping once, nesting two harmless verbose=False layers) remains
possible but does not serialize anything. 4 new tests, including a real
2-thread test proving concurrent calls now overlap instead of queuing
(previously would have taken >=0.2s serialized; now completes well under
that).

**What to expect next**: with the artificial intra-process bottleneck gone,
both `--workers` (within one shard) and `--shard-count` (across processes)
should now deliver closer to their real theoretical scaling. This also means
the *previous* 2-shard test was, without realizing it, accidentally being
kept polite to MAST by that lock -- removing it could reveal MAST's real
concurrency ceiling for the first time (2 shards x 6 workers = 12 concurrent
connections, already past `docs/SYSTEM_PROFILE.md`'s 4-6 guidance for
external-service work). The next run on version 0.2.19+ (planned as 4 shards
-- see below) should be watched closely for new errors, timeouts, or slower
per-target rates that would indicate real throttling, not assumed clean just
because the prior run was clean.

Plan at the time (2026-07-03) was to step to 4 shards next as one clean,
isolated data point before considering 6-7, per the Measure-then-scale
cadence -- superseded by the actual 4-shard and 6-shard results below, which
answered the "how far does this scale" question directly.

**4-shard and 6-shard live tests both PASS (2026-07-03/04, version 0.2.21+),
self-reported via the Run Report Policy:** 4-shard run: 1,000 targets (250
per shard, `--workers 6` each -- 24 total concurrent connections),
~2.15s/target combined -- a dramatic jump from the pre-lock-fix 17.1s/target
baseline, confirming the 0.2.19 download lock fix unlocked real intra-shard
worker concurrency, not just inter-shard. Immediately followed by a 6-shard
run: 1,500 targets (250 per shard, one row failure in one shard, `--workers
6` each -- 36 total concurrent connections), ~2.11s/target combined --
essentially flat versus the 4-shard run. Per the Measure-then-scale cadence,
this is the stop signal: 24-36 total concurrent connections (4-6 shards x 6
workers each) appears to be the practical ceiling for this workload (likely
MAST's own concurrency limit, or local network/CPU saturation, not the
previously-fixed lock), and pushing meaningfully past 36 total connections
without new evidence is unlikely to help and risks throttling. This is the
human's real standing operating cadence (6 shards x 6 workers), not a
conservative fallback -- **do not recommend a lower worker/shard count than
this without a new measurement showing it's actually necessary.**

**Concrete next step — give the human this exact recipe (version 0.2.22+ for the manifest-progress percent; 0.2.21+ minimum for the lock/WAL fixes):**

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/process_t1_kepler_batch.py --max-targets 500 --workers 6 --shard-index 0 --shard-count 6
# repeat in five more tabs with --shard-index 1, 2, 3, 4, 5 (same --shard-count 6)
```

6 shards x 6 workers (36 total connections) is the human's proven standing
cadence -- recommend it by default rather than a more conservative fallback.
All tabs share the default `--db-path`; every run's startup banner and
final "Done in Xs" line now print `n_done/n_total (XX.X%)` against the whole
6,515-target manifest, so progress is visible on every invocation without a
separate `--status-only` call. Compare the printed per-target rate against
the ~2.1-2.15s/target baseline above -- a similar-or-better rate with no new
errors is healthy; a regression or new `ERROR:`/timeout flags mean back off
shards/workers rather than pushing further. If any target shows
`NO_DATA`/`NO_LIGHTKURVE`/`ERROR:...`, that is expected for some KOIs (not
every KIC has usable long-cadence data) and is not itself a blocker unless
most targets fail. Continue with bounded invocations until the manifest
reaches 100%.

The run006/run008 notes below are historical provenance only. Preserve them so
future agents do not re-debug the same scanner failures, but do not treat them
as the next production blocker.

**Option B1–B4 is merged (PR #139).** The scanner now excludes TOI + CTOI + confirmed hosts automatically and defaults to Tmag 12–14.5. A first 200-target attempt (`logs/discovery_run_001.json`) completed on 2026-06-28 but does **not** close T1-0: it used the old SPOC-only fetch path and produced 198 no-data rows plus 2 transient remote-disconnect errors, with 0 clear scans and 0 candidates. Root cause: target selection queried TIC stars without requiring light-curve availability, while the pipeline fetched only `author='SPOC', exptime='long'`.

A second QLP attempt (`logs/discovery_run_002_qlp.json`) started on 2026-06-28 but also does **not** close T1-0: it recorded 3 errors, 0 clear scans, and 0 candidates. Root cause: interrupted prior QLP downloads left corrupt FITS files in the local Lightkurve MAST cache (`~/.lightkurve/cache/mastDownload/HLSP/...`), and the shared fetch path treated Lightkurve's "This file may be corrupt due to an interrupted download" error as a terminal scan error instead of deleting the named cache file and retrying. The next run must use a fresh log after the cache-repair fetch fix is merged.

A third QLP attempt (`logs/discovery_run_003_qlp_cache_repair.json`) started on 2026-06-28 but also does **not** close T1-0: it recorded 1 error, 0 clear scans, and 0 candidates, then crashed with `ValueError: I/O operation on closed file` while printing progress. Root cause: Lightkurve's public `SearchResult.download()` and `download_all()` are decorated with `suppress_stdout`, which mutates process-global `sys.stdout`; that is unsafe while `star_scanner.py` runs worker-thread downloads and prints progress on the main thread. The next run must use a fresh log after the shared fetch path avoids those decorated methods.

A fourth QLP attempt (`logs/discovery_run_004_qlp_stdout_safe.json`) completed on 2026-06-28 but also does **not** close T1-0: it recorded 200 total entries, 0 candidates, 0 clear scans, 1 no-data row, and 199 errors. Root cause: the shared fetcher still requested SPOC-style `pdcsap_flux` from QLP HLSP products. The downloaded FITS files were valid; QLP products contain columns such as `SAP_FLUX`, `KSPSAP_FLUX`, `DET_FLUX`, and `SYS_RM_FLUX`, but not `PDCSAP_FLUX`. Lightkurve wrapped the missing-column `KeyError('pdcsap_flux')` in the misleading "may be corrupt due to an interrupted download" message.

A fifth QLP attempt (`logs/discovery_run_005_qlp_flux_safe.json`) started after the flux-column fix but also does **not** close T1-0. The pasted console showed many Astroquery/MAST `Downloading URL ... [Done]` lines and numerical warnings but no per-target scanner progress, and no durable scan log existed before the first completed target. Root cause: `ScanLog` flushed only after `record()`, `run_background_scan()` printed only after a future completed, and Lightkurve's lower-level `_download_one()` still called `Observations.download_products()` with Astroquery's default `verbose=True`.

**PR #143 is merged (2026-06-28).** A live one-target smoke on `main` verified that the ExoFOP SSL loader, Python 3.14 helper imports, bounded TIC target selection, and no-light-curve `no_data` classification all work. Do not re-debug the old pasted failures from before PR #143.

**Run006 completed locally on 2026-06-29.** `logs/discovery_run_006_qlp_progress_safe.json` has 200 entries: 192 `candidate_found`, 6 `scanned_clear`, 1 `no_data`, 1 `error`, and 0 active targets. SHA-256: `8ed084e39fcf1b1f7f0405208a413d4651641aba195305f3ca3b2b8bc3615dc8`. `logs/discovery_filtered_006_qlp_progress_safe.json` has 2 filtered candidates. SHA-256: `17630739c28bed296910512b86c63c77d952708cf84ab2fe6d8f55ae120a5fc9`.

**Run008 targeted follow-up completed locally on 2026-06-30 after version 0.2.8 fixes.** `logs/discovery_run_008_targeted_qlp_stitch_safe.json` has 2 entries, both `candidate_found`, active `{}`, SHA-256 `8626587c4fe59565132e078273763c7beac4a0a88597615f71e147a5134d1b0a`. `logs/discovery_filtered_008_targeted_qlp_stitch_safe.json` has 2 rows, SHA-256 `574a4cf188faa9e273128496fcd23b27cb8369a3e9d2ad2c1b5bbaedd9effed4`. TIC 201252011 reproduced at P=227.39056281978395 d, FPP=0.11606180728511539. TIC 257712351 reproduced at P=142.95415231096942 d, FPP=0.12672948535351847. The earlier Lightkurve normalization warning root cause was `LightCurveCollection.stitch()` defaulting to `corrector_func=lambda x: x.normalize()` before project sigma-clipping; fetch now calls `stitch(corrector_func=None)`. Regenerated `exo --output` files now contain computed `features`; false-positive vetting no longer reports all diagnostics missing. Version 0.2.9 adds raw `diagnostics`, fetch provenance, and missing-diagnostic reasons to candidate-review outputs so reviewers can distinguish insufficient phase coverage from not-yet-run catalog/centroid checks. Version 0.2.10 retries transient MAST/Lightkurve connection disconnects during candidate packet regeneration. Both best signals still fail `limb_darkening_plausibility_score=0.0` and have many missing diagnostics, so they are **not submission-ready**.

Historical candidate rows from the superseded review loop:

| TIC | Period (d) | FPP | Pathway |
|---|---:|---:|---|
| TIC 201252011 | 227.39056281978395 | 0.1160636155807766 | `planet_hunters_discussion` |
| TIC 257712351 | 142.95415231096942 | 0.12672985673564718 | `planet_hunters_discussion` |

Treat run006 as useful scan evidence and run008 as useful targeted follow-up evidence, not submission-ready evidence. Run006 flagged 192/200 targets as candidates and 81 detections hit the 0.5 d or 500 d period boundaries; subsequent scanner code rejects invalid and period-boundary BLS peaks, so future evidence runs must use `main` at version 0.2.10 or newer. Version 0.2.10 regenerated candidate packets moved the two filtered candidates above the prior FPP < 0.15 escalation threshold, so this loop is no longer the primary production path. Do not submit/contact externally without explicit human approval.

Note (2026-07-02, version 0.2.11): a supplementary correctness fix wired real
TIC catalog stellar/contamination parameters into `vet_signal()` (see the
0.2.11 changelog entry above). This does not reopen the run006/run008 loop as
an active task — it is optional forensic curiosity only, not a required next
step. The active gap is T1-1 below.

### CNN production runbook

Use `docs/CNN_PRODUCTION_RUNBOOK.md` for the authoritative copy-paste workflow.
The correct CLI flags are `--split-dir`, `--checkpoint-dir`, and `--pretrained-checkpoint`.

Gate: raw held-out test AUC ≥ 0.85, raw held-out test F1 ≥ 0.80 (when raw ECE < 0.05,
temperature scaling is skipped and calibrated metrics equal raw metrics — see ECE-skip note above).

---

## Local-Only Data Artifacts — Read Before Any CNN Task

Large training data files are stored on the user's local Mac and are **never committed to the repository**. Before proposing or executing any CNN training task, ask the user to confirm current file state.

| File | Status | Description |
|---|---|---|
| `data/tess_snippets_v2.jsonl` | **COMPLETE** — 2,619 snippets | TESS phase-folded snippets; merged from two download runs; 56 targets had permanent MAST 404s |
| `data/kepler_snippets.jsonl` | **LOCAL VALIDATED** — 6,837 finite snippets as of 2026-06-17 | JSON parse PASS; zero non-finite flux rows; zero duplicate resume keys; split validator PASS |
| `data/kepler_cnn_splits/` | **LOCAL VALIDATED** — train 4,741 / val 1,060 / test 1,036 | Kepler split used for pretraining and as the Kepler component of `data/joint_cnn_splits/` |
| `data/tess_kepler_overlap_snippets.jsonl` | **COMPLETE** — 4,864 snippets as of 2026-06-20 | Kepler KOI stars folded at Kepler ephemerides; TESS-domain labels from KOI disposition; ~2,716 terminal failures in sidecar |
| `data/tess_combined_snippets.jsonl` | **BUILT** — 7,483 rows | Concatenation of TESS v2 + overlap; used for `data/tess_combined_cnn_splits/`; do not rebuild |
| `data/joint_cnn_splits/` | **LOCAL VALIDATED** — 9,633 train / 1,049 val / 1,033 test | C17 splits; retain for reproducibility; do NOT use as training input for future candidates — joint training caused domain mismatch |

The Kepler download uses `author="Kepler"` (prevents HLSP/IRIS cache corruption) and `socket.setdefaulttimeout(120)` (prevents WiFi-drop hangs). It resumes automatically from durable success keys plus the failure sidecar. The optimized path groups pending KOIs by `kepid`, fetches each KIC once, filters non-finite time/flux samples before phase binning, and supports polite bounded concurrency via `--workers 3 --request-delay 0.5`.

Any long-running local data pull must have durable resume state for both
successful outputs and terminal failures. Console progress is not a checkpoint.
If rerunning a downloader reprocesses completed or terminally failed work by
default, stop and fix resume state before asking the human to run it again.

**Do not assume these files are present on the agent's server.** They exist only
on the user's Mac. If the user is away from the Mac, agent-side work is limited
to runbook, validation, promotion-gate, and documentation hardening until the
human can run the local commands in `docs/CNN_PRODUCTION_RUNBOOK.md`.

## Git-Add-Safe Artifact Policy — Mandatory

The standard operator cadence is `git add .`. If that command would stage local
corpora, split outputs, checkpoints, runtime logs, generated reports, virtual
environments, rejected experiments, or cache files, the repository is wrong and
`.gitignore` must be fixed before continuing.

Other coding agents may only see GitHub. Therefore local-only artifacts must be
ignored, but their production-relevant state must be committed in
`docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json`.

When any ignored artifact affects T1-1 or another production gate, update the
ledger in the same PR as the code, runbook, or readiness change. The ledger must
answer, from GitHub alone:

1. Which local artifact paths are expected
2. Whether each artifact is missing, pending, valid, rejected, or promoted
3. The latest known counts, hashes, validation results, and approval state
4. The exact next command the human or agent should run

Do not rely on chat context, local terminal scrollback, or uncommitted files for
artifact state. A production-approved CNN checkpoint is the only CNN artifact
class that may move from ignored local state into `models/`, and only after the
evaluator passes and the human explicitly approves promotion. Because CNN model
paths are ignored defensively, approved promotion may require an intentional
`git add -f` that is documented in the promotion PR.

---

## Read First

Before writing code, recover project context from committed files. Read:

- `CLAUDE.md` — current codebase state, module map, type system, quality commands
- `docs/SCORING_MODEL.md` — mathematical specification for scoring and classification
- `docs/PIPELINE_SPEC.md` — end-to-end pipeline architecture
- `docs/PROJECT_STATUS.md` — current active state and next work
- `docs/DECISIONS.md` — durable architectural decisions
- `docs/LOCAL_ARTIFACT_LEDGER.md` — GitHub-visible state for ignored local artifacts
- `CONTRIBUTING.md` — setup, validation, and contribution policy

Do not rely on chat context, memory, or prior conversation history as the source of truth.

## MCP Server Usage

When configured and available in the agent's environment, prefer these MCP servers over guessing or ad hoc web search:

- **GitHub MCP** — issues, PRs, remote branches, repo metadata, commit/PR review, PR notes and links, branch health.
- **Context7 MCP** — current library/framework/API/CLI documentation (`resolve-library-id` then `query-docs`). Use instead of relying on training-data knowledge, since library versions and APIs change.
- **arXiv MCP** — preprint lookup, paper search, and research context for exoplanet/exomoon detection methods.
- **NASA ADS MCP** — astronomy/astrophysics literature search, bibcodes, citations, references, author metrics, and BibTeX export.

These are general-purpose research/collaboration tools, separate from this repo's project-scoped MCP servers (`exo_project_files`, `exo_git_read`, `exo_guard` — see `CLAUDE.md` "Project-Scoped MCP Servers"), which remain the sandboxed, offline-by-default servers for reading repo files, safe git inspection, and fixed validation commands. Availability of the general-purpose servers above is environment-dependent — check before assuming they're present, and fall back to `WebSearch`/`WebFetch` or manual `gh`/`git` commands if not.

## Multi-Agent Continuity

Multiple agents may work on this project across separate sessions, branches, and chat threads. Repository documentation is the continuity mechanism.

When durable instructions, architectural decisions, operating rules, or scientific assumptions are established, record them in the appropriate repository document instead of leaving them only in chat. If chat context conflicts with repository documentation, prefer repository documentation unless the user explicitly instructs otherwise in the current task.

Preserve enough rationale, provenance, and test evidence in commits, docs, and code comments for another agent to continue without needing the conversation that produced the change.

## Branch And Git Policy

Default development should happen on a non-`main` branch and be merged through review. Do not push directly to `main` unless the current user explicitly requests a direct `main` commit or push.

Before committing, check `git status --short --branch`. Do not overwrite or revert unrelated user changes.

## Quality Gates

Run these before every commit when the local environment supports them:

```bash
ruff check .
python -m mypy src
PYTHONPATH=src python -m pytest
```

If a gate cannot run because of a local environment issue, record the exact blocker in the handoff or commit message. Default tests must not require live external services.

## Code Standards

- Python 3.14.3 (validated runtime; minimum acceptable is 3.11).
- Use `from __future__ import annotations` in Python modules.
- Prefer Pydantic v2 frozen models for structured data contracts.
- Public functions must be fully typed.
- `OptScore = float | None`: absent diagnostics contribute neutrally to log scores, while threshold gates treat missing participating diagnostics conservatively.
- Add comments for why, not for obvious what.
- Keep changes scoped to the task and existing architecture.

## Testing Standards

Every meaningful code change needs appropriate tests. Required coverage should scale with risk:

- Unit tests for public functions and numerical thresholds
- Fixture-based tests for complex interactions
- `None`-input tests for `OptScore` paths
- Conservation and sanity tests for posteriors and bounded scores
- Integration tests for pipeline behavior with mocked external services
- `@pytest.mark.integration_live` for tests requiring live network access

Do not include live service tests in the default suite.

## Scientific Guardrails

Follow `docs/SCORING_MODEL.md` guardrails:

- Never emit "confirmed planet" for internally detected signals.
- Use "candidate signal", "possible transit-like event", or "follow-up target".
- Always expose false-positive evidence.
- Preserve provenance for scores, thresholds, inputs, and generated reports.
- Suppress formal submission pathways if key diagnostics are missing.
- Prefer conservative classifications over optimistic ones.
- External submission or contact requires explicit human approval.

## Background Automation

Background search automation uses top-level configuration and top-level SQLite runtime logs:

```text
configs/background_search_v0.json
logs/background_search.sqlite3
```

Generated SQLite databases and background report exports are runtime artifacts. Do not commit them unless a future decision explicitly promotes a fixture artifact.

The authoritative one-shot command is:

```bash
exo background-run-once
```

Schedulers should call one bounded run at a time, capture stdout/stderr, and avoid overlaps. See `docs/BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`, `docs/BACKGROUND_SEARCH_SQLITE_SCHEMA.md`, and `docs/SCHEDULER.md`.

## Local System Profile

`docs/SYSTEM_PROFILE.md` is a committed production directive and the
authoritative local hardware profile for this project. It must remain in the
repository and must not be treated as a disposable local note or untracked
artifact.

Before performance-sensitive changes, large jobs, worker-count defaults,
batch-size defaults, cache layout changes, or long-running user recipes, read
`docs/SYSTEM_PROFILE.md` and optimize project defaults for the recorded MacBook
Pro M4 Max profile. Keep the scientific code portable and configurable: do not
hardcode local machine assumptions into candidate detection, scoring,
classification, or pathway logic. If code needs machine-specific behavior,
expose it through configuration, CLI flags, or documented runtime defaults.

AI/ML training code must prefer local acceleration by default. For PyTorch
training, use a configurable `device=auto` policy that selects Apple Metal/MPS
on the recorded M4 Max when available, then CUDA when available, and falls back
to CPU only when no accelerator is available or the operator explicitly selects
CPU. Startup banners for training runs must print the resolved device so the
operator can tell whether the GPU is actually in use.

Other performance-sensitive code should use bounded parallelism when it is
scientifically safe and operationally useful. Prefer multiprocessing or
multithreading over strictly serial loops for CPU-local batch work, starting
near the worker counts in `docs/SYSTEM_PROFILE.md`. Live external-service
workloads must remain polite and bounded; never use unbounded concurrency or a
worker default that risks throttling MAST, ExoFOP, NASA Exoplanet Archive, or
similar services. Every parallel default must remain configurable.

## macOS Long-Running Process Policy

Any Python command expected to run longer than ~60 seconds **must** be prefixed with `caffeinate -i` in recipes given to the user. This prevents macOS from sleeping and killing the process mid-run.

```bash
# Standard form for any long download or training run:
caffeinate -i .venv/bin/python Skills/<script>.py [args]

# To keep running with lid closed, use -dims instead:
caffeinate -dims .venv/bin/python Skills/<script>.py [args]
```

This applies to: light curve downloads, CNN training, batch scans, injection-recovery runs, and any other script that makes repeated network calls or runs for more than a minute. Never give a bare `python ...` recipe for these — always prepend `caffeinate -i`.

## Console Output and ETA — MANDATORY

**Every script that iterates over N items or trains for N epochs must print real-time progress.**
The operator cannot see internal state; silent scripts look identical to hung ones.

### Required pattern for item loops

```python
import time

start = time.monotonic()
for i, item in enumerate(items, 1):
    # ... do work ...
    elapsed = time.monotonic() - start
    rate = i / elapsed
    remaining = (n_total - i) / rate if rate > 0 else float("inf")
    eta = f"{remaining/60:.0f}m{remaining%60:.0f}s" if remaining > 90 else f"{remaining:.0f}s"
    print(f"  [{i}/{n_total}]  elapsed={elapsed:.0f}s  ETA={eta}", flush=True)
```

### Required pattern for training loops

Print one line per epoch with at minimum: epoch number, train loss, val loss, primary metric, learning rate, and whether this is a new best or how far patience has advanced:

```
Epoch  N/50  train=0.4123  val=0.5210  auc=0.8011  lr=3.00e-04  ← best
Epoch  N/50  train=0.3990  val=0.5350  auc=0.7944  lr=3.00e-04  (patience 1/10)
```

Print a startup banner before the loop showing total size, batch size, max epochs, and patience.
Print an explicit early-stopping or completion line at the end.

### Non-negotiable rules

- Always `flush=True` on every progress print — buffered output defeats the purpose.
- Print at every step, or at minimum every 10 items for very fast loops.
- **Never commit a long-running script that has no console output** — if a script is silent, add progress prints before committing.
- When reviewing or modifying any existing long-running script, verify it meets this standard and add output if missing.

## Local–Remote Sync Policy

The user's local Mac and GitHub `main` are the joint source of truth. The agent's server environment is a temporary workspace only. Keeping them in sync is a hard requirement — never leave them diverged.

### Rules for the agent

1. **All code changes must reach `main` before the user runs anything.** The full cycle is mandatory: feature branch → commit → push → PR → CI green → merge to main → PR closed. Never leave a PR open at the end of a session.
2. **Never tell the user to run a script that has not yet been merged to main.** If a script is still on a feature branch, merge it first.
3. **Every recipe given to the user must begin by switching to `main` and fast-forwarding from `origin/main`** so their local is guaranteed current before any command executes and feature branches cannot receive divergent pulls.
4. **PRs must be merged, not just approved.** After CI passes, promote from draft, squash-merge, and confirm the PR is closed before the session ends.
5. **After every merge**, remind the user to run `git switch main` and `git pull --ff-only origin main` on their Mac if they have a terminal open.

### Standard recipe header (copy-paste this before every user command)

```bash
# Always sync first
git switch main
git pull --ff-only origin main
```

For long-running commands, the full header is:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/<script>.py [args]
```

### What Not To Do

- Do not tell the user to run `python Skills/foo.py` before `foo.py` is on `main`.
- Do not leave PRs in draft or open state at end of a session.
- Do not commit directly to `main` — always use a feature branch and PR.
- Do not assume the user's local is current — always prepend the branch-safe `git switch main` + `git pull --ff-only origin main` sync block.

## Python Environment Policy

This project runs inside a `.venv` virtual environment. **Never touch or run system Python.**

- All `pip install` commands must be run with the venv active (`(.venv)` in the prompt)
- Never run `/Applications/Python*/Install\ Certificates.command` — this modifies system Python
- Never suggest `sudo pip install` or `pip install --system`
- Never reference `/Library/Frameworks/Python.framework/` paths — those are system Python
- If an SSL or package issue arises, fix it inside the venv: `pip install <package>` with venv active is always safe and venv-scoped
- To verify the venv is active before suggesting any pip/python commands, check that the prompt starts with `(.venv)`

## What Not To Do

- Do not add features, abstractions, or refactors beyond what the task requires.
- Do not skip validation silently.
- Do not claim a signal is a confirmed planet.
- Do not enable live network access in default tests.
- Do not hide durable rules in chat-only context.
- Do not touch system Python — all Python work happens inside the `.venv`.
