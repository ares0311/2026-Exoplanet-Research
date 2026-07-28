# PROJECT STATUS

## Status: EXO-Hunter 0.5.0 PROD Accepted
## Last Updated: 2026-07-27 (version 0.5.0)

The current contract adds the persistent `ExoHunter` slash terminal that the
0.4.0 acceptance did not contain. Implementation, real new/follow-up
create-through-execution workflows, all PR #322 CI runs, squash merge
`6b8c78e`, merged-main CI, synchronization, installed slash discovery, and
clean 10/10 merged-main local gates with 3,212 tests pass. Current evidence:
`artifacts/manifests/hunter_live_acceptance_v13.json`.

This file had drifted badly stale before this update: it described Phase 4
individual-transit diagnostics and T1-2 stacking calibration as the active
work, when both closed many versions earlier (Phase 4 closed at 0.2.81;
T1-2 closed well before that). Per the PRIMARY DIRECTIVE in `AGENTS.md`,
do not trust a cached snapshot of "current work" here or anywhere else —
`docs/PRODUCTION_READINESS.md` is the single actively-maintained
version-by-version changelog and is re-read fresh every session. This file
now points to it rather than duplicating it, to avoid re-accumulating the
same drift.

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- A merged-main, live-accepted EXO-Hunter lifecycle from a 10,000-candidate
  universe through immutable search execution, append-only results/provenance,
  and evidence-gated follow-up recommendations; see
  `artifacts/manifests/hunter_live_acceptance_v1.json`
- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify, with an
  animated terminal presentation (Milestone 20, version 0.2.96) that is purely
  cosmetic and auto-disables under redirected output/CI
- Bayesian log-score model over six hypotheses
- XGBoost, stacking, CNN (`benchmark_cnn_v1`, promoted), and full-ensemble
  scorer modes — all `PRODUCTION READY` per `docs/PRODUCTION_READINESS.md`'s
  Live-Readiness Summary
- SQLite-backed background automation with top-level logs
- 128 standalone `Skills/` Python files as of 2026-07-17 (re-run
  `rg --files Skills -g '*.py' | wc -l` for the current count — this number
  drifts and should not be trusted beyond the date above)
- 155 top-level test files, 3,051 default tests passing (exact current
  figure: `docs/PRODUCTION_READINESS.md`'s "Test baseline" line)
- 34 package Python modules under `src/exo_toolkit/`
- JWST time-series ingestion wired into the CLI with `--mission JWST`
- Novel TESS target scanning that excludes TOI, CTOI, and confirmed-host catalogs
- AGENTS.md Rule 7 (Run Report Policy) is retrofitted across all 15 tracked
  acquisition/processing Skills as of version 0.2.95

Local validation note: validated on Python 3.14.3 in `.venv` with `xgboost` dependency restored and macOS OpenMP runtime (`libomp`) installed. System Python is never used.

**The version-by-version changelog previously duplicated here (versions
0.2.55 through 0.2.77) has been removed** in favor of a single pointer, to
stop this file from re-accumulating drift the way it did before this
2026-07-17 correction. Read `CLAUDE.md`'s changelog narrative for the full,
current, version-by-version history — it now runs through version 0.2.96
and covers everything this file used to duplicate plus everything since:
the completed Phase 3 representation-learning tracks (both branches closed
`no_external_added_value`/evidence-limited, training not authorized), the
completed Phase 4 individual-transit-detector extension (all three named
increments), the completed AGENTS.md Rule 7 Run Report Policy retrofit
(all 15 tracked scripts), and the completed Milestone 20 animated CLI.
`docs/PRODUCTION_READINESS.md` carries the equivalent narrative with the
Tier 1/Tier 2 gap framing; `docs/ROADMAP.md` carries it with the
milestone/phase framing. All three should agree; if they ever don't,
treat that disagreement itself as a defect to fix, not a call to guess
which one is right.

---

## What Is Complete

| Area | Key Files | Status |
|------|-----------|--------|
| Scoring engine | `schemas.py`, `features.py`, `hypotheses.py`, `priors.py`, `scoring.py`, `pathway.py` | Complete |
| Data pipeline | `fetch.py`, `clean.py`, `search.py`, `vet.py`, `calibration.py` | Complete |
| Transit scan CLI | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output`; JWST via `--mission JWST`; animated/plain progress display (Milestone 20) | Complete |
| Background automation CLI | `cli.py` — `background-run-once`, summaries, integrity, validation | Complete |
| Background automation module | `background/` — config, fixtures, priority, runner, storage, reports | Complete |
| EXO-Hunter lifecycle | `search_lifecycle.py`, `hunter_cli.py` — exact new/follow-up manifests, resumable runs, append-only history, follow-up registry | PROD accepted on merged-main live data |
| SQLite runtime state | `logs/background_search.sqlite3` schema v2 | Complete |
| Background config | `configs/background_search_v0.json` | Complete |
| Scoring prior config | `configs/scoring_priors_v0.json` — conservative default plus TESS/Kepler/K2 profiles | Complete |
| Scheduler docs | `docs/SCHEDULER.md` — cron, launchd, systemd | Complete |
| ML Tier 1 | `ml/xgboost_scorer.py` + `models/xgboost_koi.json` | Complete — trained on 7,586 Kepler KOIs (AUC=0.992) |
| ML Tier 2 | `ml/cnn_scorer.py` + `models/cnn/benchmark_cnn_v1/` | Complete — promoted checkpoint, test AUC 0.9572 |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete, calibrated full-ensemble weights (XGBoost=0.95/CNN=0.00/Bayesian=0.05) wired into `cli.py` |
| Training/evaluation Skills | Kepler, TESS, combined training, CNN data assembly/validation/training support, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete; run006/run008 evidence retained as historical provenance |
| Additional Skills | Analysis, vetting, observability, ML, physics, reporting, scheduling, and follow-up utilities retained after production-scope cleanup | Complete |
| Milestone 19a-d Skills | Multi-sector phase compare, static dashboard, local read-only API, interactive browser | Complete |
| Milestone 20 | Animated terminal presentation for `exo scan`, `--no-animation`, interruption handling | Complete (version 0.2.96) |
| Milestone 30 Skills | 15 diagnostics + scheduling tools including `flux_anomaly_detector`, `candidate_confidence_tracker`, `uncertainty_propagator`, `multi_target_scheduler`, `candidate_archive`, and 10 more | Complete |
| Milestones 34-39 Skills | 90 additional ML evaluation, photometry quality, transit vetting, noise budget, orbit simulation, stellar physics, TTV, occurrence-rate, and planning utilities | Complete |
| Phase 3 representation learning | External foundation-model baseline (Chronos-Bolt tiny, Astromer2) + stellar-variability/injection benchmarks | Complete — `no_external_added_value`; frozen CNN remains strongest comparator; training not authorized |
| Phase 4 individual-transit diagnostics | Per-transit duration/midpoint, missing-transit-fraction, transit-asymmetry, extra-event-count scores | Complete — all three named extensions wired into `log_score_planet()`/`log_score_instrumental()` |
| AGENTS.md Rule 7 retrofit | Run Report Policy across all 15 tracked acquisition/processing Skills | Complete (version 0.2.95) |
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

**None.** No Tier 1 or Tier 2 gap is open: T1-0/T1-1/T1-2 are all complete
(`benchmark_cnn_v1` is the promoted CNN checkpoint; stacking calibration is
wired into `cli.py` with weights XGBoost=0.95/CNN=0.00/Bayesian=0.05). All
scorer modes (`bayesian`, `xgboost`, `ensemble`, `cnn`, `full-ensemble`) are
`PRODUCTION READY` per `docs/PRODUCTION_READINESS.md`'s Live-Readiness
Summary. Phase 3 representation learning, Phase 4 individual-transit
diagnostics, the AGENTS.md Rule 7 Run Report Policy retrofit, and Milestone
20's animated CLI are all closed as of version 0.2.96.

Per the PRIMARY DIRECTIVE in `AGENTS.md`: a closed gap list does not mean
work should stop or that the project is live in production. The correct
process each session is to read `docs/PRODUCTION_READINESS.md`,
`docs/ROADMAP.md`, and `docs/DISCOVERY_RUNBOOK.md` fresh, then choose the
highest-impact task from the pre-deployment checklist, open production
defects, or real operator workflow gaps — not to assume a hardcoded "next
action" from this file, which is exactly the kind of stale pointer that
caused this file's previous drift.

`docs/DISCOVERY_RUNBOOK.md`'s live-search follow-up (TIC 355651994, P=97.16
d) is evidence-exhausted given currently available data: neither of its two
independent TESS-SPOC target-pixel products covers the predicted event, and
the QLP light curve has only two independent events versus the four required
for odd/even testing. Resolving it further requires new observations
(a human/telescope-time blocker), not more code.

## Historical Discovery Evidence

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
- Version 0.2.9 adds candidate-review diagnostics: `exo --output` now
  serializes raw vetting diagnostics, fetch provenance, and missing-feature
  names, and `Skills/false_positive_vetter.py` explains why missing
  diagnostics are unavailable.
- Version 0.2.10 adds bounded retry/backoff for transient MAST/Lightkurve
  connection disconnects during candidate-review packet regeneration.
- Run008 targeted follow-up reproduced both filtered candidates under the fixed
path: `logs/discovery_run_008_targeted_qlp_stitch_safe.json` has 2
`candidate_found` entries and active `{}`. SHA-256:
`8626587c4fe59565132e078273763c7beac4a0a88597615f71e147a5134d1b0a`.
Filtered output SHA-256:
`574a4cf188faa9e273128496fcd23b27cb8369a3e9d2ad2c1b5bbaedd9effed4`.

This loop is now historical. It is useful scanner evidence, but later v0.2.10
candidate packet regeneration moved the two filtered candidates above the prior
FPP < 0.15 escalation threshold. Do not submit/contact externally from this
evidence without explicit human approval, and do not use it to block T1-1.

## Trained-Model History

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
- Source access smoke test passed end-to-end on 2026-07-02 with TAP schemas/rows,
  ExoFOP CSV, and Lightkurve Kepler/TESS searches verified.
- Storage/source snapshot planning passed on 2026-07-02: committed metadata
  records source row counts, sample MAST product metadata, and an under-cap
  92,093,823,360-byte combined Kepler-long-cadence plus TESS estimate.
- The full T1-1 outcome (checkpoint training, promotion, calibration) and
  everything since is in `CLAUDE.md`'s changelog narrative — not repeated
  here to avoid re-duplicating the drift this file just had corrected.

---

## Next Actions

There is no single hardcoded "next action" — the PRIMARY DIRECTIVE in
`AGENTS.md` requires reading `docs/PRODUCTION_READINESS.md`,
`docs/ROADMAP.md`, and `docs/DISCOVERY_RUNBOOK.md` fresh each session and
choosing the highest-impact task from current evidence, not from a stale
list here (this exact anti-pattern is what left this file describing T1-2
as pending for dozens of versions after it actually closed). As of version
0.2.96:

- No Tier 1/Tier 2 gap is open.
- `docs/ROADMAP.md`'s "Current Master-Guide Alignment" priority list is
  fully closed through Milestone 20.
- The one open live-search follow-up (TIC 355651994) is evidence-exhausted
  pending new telescope observations — a human/hardware blocker, not a
  coding task.
- Before starting new work, re-derive the next task from current readiness
  checks, roadmap state, and any newly reported defect, per the PRIMARY
  DIRECTIVE's impact ordering (production blocker > failing check > roadmap
  item > validation/reliability/operability improvement).

Remote sync note: local `main` is synced with `origin/main` as of the latest
handoff.

## Latest Local Validation

See `docs/PRODUCTION_READINESS.md`'s "Test baseline" line for the current
verified figure (re-run `.venv/bin/python Skills/run_quality_gates.py` to
reproduce) — not hardcoded here, since a hardcoded number here is exactly
what went stale before this correction.

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
