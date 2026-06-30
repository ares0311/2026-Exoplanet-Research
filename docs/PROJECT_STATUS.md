# PROJECT STATUS

## Status: Active Development
## Phase: Live Discovery Gate — run008 targeted QLP false-positive review
## Last Updated: 2026-06-30

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes (Tier 1 model trained: Kepler KOI AUC=0.992)
- SQLite-backed background automation with top-level logs
- 98 standalone `Skills/` utility scripts (`rg --files Skills -g '*.py'`)
- 114 top-level test files
- 27 package Python modules under `src/exo_toolkit/`
- JWST time-series ingestion wired into the CLI with `--mission JWST`
- Novel TESS target scanning that excludes TOI, CTOI, and confirmed-host catalogs

Local validation note: validated on Python 3.14.3 in `.venv` with `xgboost` dependency restored and macOS OpenMP runtime (`libomp`) installed. System Python is never used.

---

## What Is Complete

| Area | Key Files | Status |
|------|-----------|--------|
| Scoring engine | `schemas.py`, `features.py`, `hypotheses.py`, `priors.py`, `scoring.py`, `pathway.py` | Complete |
| Data pipeline | `fetch.py`, `clean.py`, `search.py`, `vet.py`, `calibration.py` | Complete |
| Transit scan CLI | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output`; JWST via `--mission JWST` | Complete |
| Background automation CLI | `cli.py` — `background-run-once`, summaries, integrity, validation | Complete |
| Background automation module | `background/` — config, fixtures, priority, runner, storage, reports | Complete |
| SQLite runtime state | `logs/background_search.sqlite3` schema v2 | Complete |
| Background config | `configs/background_search_v0.json` | Complete |
| Scoring prior config | `configs/scoring_priors_v0.json` — conservative default plus TESS/Kepler/K2 profiles | Complete |
| Scheduler docs | `docs/SCHEDULER.md` — cron, launchd, systemd | Complete |
| ML Tier 1 | `ml/xgboost_scorer.py` + `models/xgboost_koi.json` | Complete — trained on 7,586 Kepler KOIs (AUC=0.992) |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete, includes optional 3-tier XGBoost/CNN/Bayesian blend |
| ML Tier 2 scaffolding | `ml/cnn_scorer.py`, `Skills/train_cnn.py`, CNN data utilities | Complete, no production checkpoint; paused until discovery scan evidence exists |
| Training/evaluation Skills | Kepler, TESS, combined training, CNN data assembly/validation/training support, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete; first real QLP scan complete and review-blocked |
| Additional Skills | Analysis, vetting, observability, ML, physics, reporting, scheduling, and follow-up utilities retained after production-scope cleanup | Complete |
| Milestone 19a Skill | `multi_sector_phase_compare.py` — offline per-sector phase-fold comparison | Complete |
| Milestone 19b Skill | `candidate_dashboard_export.py` — static conservative candidate dashboard with optional plot artifacts | Complete |
| Milestone 19c Skill | `candidate_api.py` — local read-only candidate API plus optional background SQLite summaries | Complete |
| Milestone 19d Skill | `candidate_browser_ui.py` — interactive local candidate browser with optional plot previews | Complete |
| Milestone 30 Skills | 15 diagnostics + scheduling tools including `flux_anomaly_detector`, `candidate_confidence_tracker`, `uncertainty_propagator`, `multi_target_scheduler`, `candidate_archive`, and 10 more | Complete |
| Milestones 34-39 Skills | 90 additional ML evaluation, photometry quality, transit vetting, noise budget, orbit simulation, stellar physics, TTV, occurrence-rate, and planning utilities | Complete |
| CTOI source contract | `docs/CTOI_SOURCE_CONTRACT.md`, `Skills/fetch_exofop_ctoi.py`, `tests/fixtures/exofop_ctoi_sample.csv`, `tests/fixtures/exofop_ctoi_labels_sample.json` — opt-in fixture-backed community candidate labels | Complete, excluded from default training |
| Project MCP bootstrap | `.mcp.json`, `.codex/config.toml`, `Skills/mcp_bootstrap_server.py` — project-scoped file, git-read, and fixed validation MCP servers | Complete, offline by default |
| Live label-check audit | `Skills/count_tess_labels.py`, `Skills/tess_label_check_summary.py` — opt-in live ExoFOP gate check plus read-only SQLite log summary | Complete, live access requires intentional approval |
| Docs | `README.md`, `docs/`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md` | Active maintenance |

---

## Background Automation State

The implementation follows `docs/BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`:

- Single bounded command: `exo background-run-once`
- Top-level SQLite logs under `logs/`
- Top-level config under `configs/`
- Known TESS fixtures plus synthetic edge cases
- Composite target priority with exposed component factors and reason codes
- Exactly-one primary outcome target: reviewed or needs-follow-up
- Mandatory follow-up test records for escalated targets
- Conservative draft reports exported as Markdown and HTML
- Ranked top-three submission recommendations
- Explicit human-approval records; no external submission command is implemented
- Scheduler-friendly exit codes behind `--scheduler-exit-codes`
- Non-overlap locking with a brief wait

Runtime artifacts remain ignored by default:

```text
logs/background_search.sqlite3
reports/
checkpoints/
data/*.jsonl
data/*_splits/
models/cnn*/
```

Production-relevant ignored artifact state is committed in
`docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json` so GitHub-only agents can continue
without relying on chat context or local terminal output.

---

## Active Production Blocker

**T1-0 — first real discovery scan evidence**

- The project mission has been realigned to discovering previously unknown transit candidates before doing more CNN work.
- Option A JWST integration is merged: A1/A2 via PR #133 and A3 CLI wiring via PR #141.
- Option B TESS novelty targeting is merged via PR #139: `star_scanner.py` excludes TOI, CTOI, and confirmed-host catalogs, and defaults to Tmag 12.0-14.5.
- Live scanner startup/target-selection hardening is merged via PR #143: ExoFOP SSL loading, Python 3.14 helper imports, bounded TIC target selection, and `no_data` classification are fixed; a one-target live smoke on `main` selected TIC 425884922 and completed with `1 no-data | 0 errors`.
- K2 overlap corpus collection is complete locally with 2,086 snippets; do not re-fetch it.
- A SPOC-only 200-target attempt completed on 2026-06-28 as `logs/discovery_run_001.json`, but it did not close T1-0: 198 targets had no SPOC long-cadence light curve, 2 hit transient remote disconnects, and there were 0 clear scans / 0 candidates.
- A QLP attempt started on 2026-06-28 as `logs/discovery_run_002_qlp.json`, but it did not close T1-0: it recorded 3 corrupt local Lightkurve cache errors, 0 clear scans, and 0 candidates. Root cause: interrupted prior QLP downloads left corrupt FITS files under `~/.lightkurve/cache/mastDownload/HLSP/...`, and the shared fetch path did not delete the named corrupt file and retry.
- A cache-repair QLP attempt started on 2026-06-28 as `logs/discovery_run_003_qlp_cache_repair.json`, but it did not close T1-0: it recorded 1 error, 0 clear scans, and 0 candidates before crashing with `ValueError: I/O operation on closed file`. Root cause: Lightkurve public download methods use `suppress_stdout`, which mutates process-global `sys.stdout`; that is unsafe under worker-thread downloads while the main thread prints progress.
- A stdout-safe QLP attempt completed on 2026-06-28 as `logs/discovery_run_004_qlp_stdout_safe.json`, but it did not close T1-0: it recorded 200 total entries, 0 candidates, 0 clear scans, 1 no-data row, and 199 errors. Root cause: the shared fetch path requested SPOC-style `pdcsap_flux`; valid QLP HLSP products provide `KSPSAP_FLUX`, `DET_FLUX`, `SYS_RM_FLUX`, or `SAP_FLUX`, not `PDCSAP_FLUX`.
- A flux-safe QLP attempt started as `logs/discovery_run_005_qlp_flux_safe.json`, but it did not close T1-0: the pasted console showed third-party MAST download chatter and warnings but no per-target scanner progress, and no durable log existed before the first completed target.
- PR #150 is merged on `main`: scanner logs are created immediately, active targets are checkpointed separately, per-target startup/progress is printed, and third-party MAST download banners are suppressed.
- Run006 completed locally on 2026-06-29 and produced durable scan evidence:
200 entries, 192 `candidate_found`, 6 `scanned_clear`, 1 `no_data`, 1 `error`,
and 0 active targets. The filtered output contains two rows: TIC 201252011
(period 227.39056281978395 d, FPP 0.1160636155807766) and TIC 257712351
(period 142.95415231096942 d, FPP 0.12672985673564718).
- Version 0.2.8 fixes two production-review blockers found during targeted
follow-up: Lightkurve stitch no longer normalizes QLP products before project
sigma-clipping, and `exo --output` now serializes computed vetting features for
`Skills/false_positive_vetter.py`.
- Run008 targeted follow-up reproduced both filtered candidates under the fixed
path: `logs/discovery_run_008_targeted_qlp_stitch_safe.json` has 2
`candidate_found` entries and active `{}`. SHA-256:
`8626587c4fe59565132e078273763c7beac4a0a88597615f71e147a5134d1b0a`.
Filtered output SHA-256:
`574a4cf188faa9e273128496fcd23b27cb8369a3e9d2ad2c1b5bbaedd9effed4`.

The next action is false-positive review, not rerun: inspect the two targeted
candidates and the missing diagnostics before any external action. Both best
signals still fail `limb_darkening_plausibility_score=0.0`, and many
centroid/contamination/odd-even/multi-sector diagnostics remain unavailable.

## Paused

**Production ML Tier 2 — checkpoint generalization**

- The label gate is open, but no CNN checkpoint has passed the production gate.
- The first seed-42 checkpoint completed training but was rejected on 2026-06-10.
- Held-out test AUC was 0.7404 and calibrated F1 was 0.6297, below the documented 0.85 and 0.80 targets.
- Validation-fitted Platt calibration worsened test Brier score and ECE, so no calibration or checkpoint artifact was promoted into `models/`.
- A 2026-06-10 audit found that every nominally usable snippet had `epoch_bjd=0.0`, so catalog transit events were not centered in phase.
- The old corpus, original seed-42 split, and temporary replacement split are retired.
- The local TESS v2 corpus is complete.
- The pre-fix local Kepler corpus was rejected on 2026-06-17: it had 7,454 rows, but 7,132 rows contained non-finite flux and only 322 finite examples survived builder filtering.
- The rebuilt local Kepler corpus was validated on 2026-06-17: 6,837 parseable finite snippets, zero duplicate resume keys, labels negative=4,280 and positive=2,557.
- `data/kepler_cnn_splits` passed validation on 2026-06-17 with train/val/test = 4,741 / 1,060 / 1,036.
- Tiny corrupt Kepler Lightkurve cache files were quarantined locally before training resumed.
- Kepler pretraining completed locally on 2026-06-18:
  `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256
  `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`;
  startup banner `device=mps`; best epoch 19, best validation loss 0.3905,
  best validation AUC 0.9186.
- `data/tess_cnn_splits` passed validation on 2026-06-18 with total examples
  2,110 and train/val/test = 1,477 / 318 / 315.
- Kepler->TESS fine-tuning completed locally on 2026-06-18 using
  `device=mps`; `checkpoints/cnn_tess_finetuned/best.pt` SHA-256
  `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`;
  best epoch 22, best validation AUC 0.8408.
- Production evaluation rejected that fine-tuned checkpoint: raw test AUC
  0.8115, raw test F1 0.7523, calibrated test F1 0.7508, calibrated Brier
  0.1966, and calibrated ECE 0.1152. It must not be promoted into `models/`.
- Path A TESS expansion was approved on 2026-06-18 as the next T1-1 strategy
  and started with an ExoFOP TOI/CTOI inventory against
  `data/tess_snippets_v2.jsonl`.
- The first Path A inventory completed locally on 2026-06-18 and found only 56
  new labeled TIC IDs (16 positive, 40 negative). This is too small to justify
  a long MAST fetch or candidate-12 training as a production-closing attempt.
- The historical ExoMAST TESS TCE endpoint used by `Skills/tess_tce_fetcher.py`
  returned HTTP 404 on 2026-06-18. The helper now reports `Flag: UNAVAILABLE`
  instead of hiding the stale provider behind a generic invalid/empty result.
- Architecture details: `docs/CNN_SPEC.md`.
- Human local runbook: `docs/CNN_PRODUCTION_RUNBOOK.md`.
- Next outside blocker: wait for first discovery-scan evidence. Do not build C20 or train another CNN until the scan is complete and reviewed.

---

## Next Actions

1. Review `logs/discovery_run_008_targeted_qlp_stitch_safe.json`, ranked candidates, and `logs/discovery_filtered_008_targeted_qlp_stitch_safe.json` as the current targeted evidence.
2. Use the regenerated `exo --output` rows and false-positive vetting reports for TIC 201252011 and TIC 257712351; both best signals remain review-blocked by missing diagnostics and failed limb-darkening plausibility.
3. Add or run candidate-specific centroid, contamination, odd/even, and multi-sector diagnostics before any external action.
4. Investigate why run006 flagged 192/200 targets as candidates and why many detections hit the 0.5 d / 500 d period boundaries before another blind scan.
5. Do not run C20 CNN corpus assembly or training until the first discovery scan is reviewed.
6. Promote nothing unless a future evaluator run reports `Flag: PASS`, raw test
   AUC is at least 0.85, calibrated test F1 is at least 0.80, calibrated
   Brier/ECE are no worse than raw, and the human explicitly approves
   promotion.

Remote sync note: local `main` is synced with `origin/main` as of the latest
handoff.


## Latest Local Validation

Validated on 2026-06-30:

```bash
.venv/bin/ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

Result: ruff passed, mypy passed, pytest passed with 2,355 passed, 2 deselected, and 3 package warnings.

---

## Key Design Decisions In Effect

- Bayesian log-score model is default; XGBoost and ensemble are opt-in.
- `OptScore = float | None`: missing diagnostics contribute neutrally to log scores.
- Missing diagnostics fail threshold gates conservatively.
- Conservative priors keep false positives prominent; mission-specific prior profiles are opt-in through `configs/scoring_priors_v0.json`.
- `provenance_score` is computed from cadence, sector count, and pipeline quality.
- `toi_checker.py` should be consulted before investing pipeline time on any new target.
- Default tests must mock external services; live tests require `integration_live`.
- **Never output "confirmed planet"** — always "candidate signal" or "follow-up target". This is a hard guardrail, not a preference.
- Background automation uses SQLite for durable state and deterministic fixtures by default.
- Background automation obeys the human-approval gate; no external submission without review.

---

## Project Mission (clarified 2026-06-22)

This project **identifies transit candidates for human review**. It does not confirm discoveries and makes no discovery claims.

The intended workflow is:

```
Pipeline output (candidates + FPP + pathway)
    → Human review process (separate, owner-defined)
        → Submission of vetted candidates only
```

The pipeline's role ends at outputting a ranked candidate list with supporting diagnostics. What happens next — reviewing phase-fold plots, checking against known catalogs, consulting experts, deciding what to submit — is entirely outside this codebase and is the human operator's responsibility.

This framing matters for how we evaluate data quality and model performance:
- **TESS SPOC photometry** is the same public data professional astronomers use for transit detection. It is appropriate for identifying transit-like signals.
- **It is not sufficient for confirmation.** Confirmation requires RV measurements, high-resolution imaging, and/or spectroscopy — none of which this pipeline provides or implies.
- The CNN model's role is to reduce false positives in the candidate list, not to confirm planets. An AUC of 0.85 means the model ranks 85% of real transits above 85% of false positives. It is a triage tool.
- All submission pathway labels (`tfop_ready`, `planet_hunters_discussion`, etc.) describe what kind of follow-up is appropriate, not what has been found.
