# Data Selection Decision Log

## 2026-07-13 — Cached TESS representation preprocessing benchmark v1

**Repo:** 2026 Exoplanet Research
**Data:** Deterministic 36-product/36-TIC sample from the committed 11,960-row
`tess_cached_unlabeled_representation_v1` training inventory, spanning TESS
sectors 1-98.
**Role:** Training only; role and leakage restrictions are unchanged.
**Execution:** Six supervised Python shard subprocesses with six cached-FITS
workers each; zero archive queries, zero downloads, and zero persisted derived
arrays.
**Preprocessing:** Finite `TIME`/`PDCSAP_FLUX`, `QUALITY == 0`, per-product
median/robust-MAD normalization, and linear resampling to 2,048 float32 flux
bins.
**Measured outcome:** 36/36 products succeeded with zero failures in 0.4197
seconds of measured preprocessing (85.77 products/s). The sample read 81.85 MB
and 807,636 cadences, retaining 704,704. Each normalized flux vector is 8,192
bytes.
**Projection:** The full 11,960-product normalized-flux-only footprint is
97,976,320 bytes (0.09797632 GB), with 139.44 seconds projected at the observed
aggregate rate. Maximum observed shard-process high-water RSS was 83,591,168
bytes.
**Storage decision:** Prefer streaming/on-demand preprocessing because the
entire inventory projects to about 2.3 minutes and no derived-array persistence
is needed for acceptable throughput. This benchmark does not authorize model
training or move training arrays into the Dropbox repository while the shared
Lightkurve cache remains above the 80 GB caution threshold.
**Evidence:**
`artifacts/manifests/representation_preprocessing_benchmark_v1.json`, SHA-256
`08c68fca194a6a5f515a17ab7e667cfd453d8f59abd8ba7bfccb13fcc207fa49`;
Run Report commit `9f1b9e8`.
**Next action:** Design a materially broader Phase 3 experiment that streams
this training-only source and adds the still-required stellar-variability
labels, injection-recovery comparison, and external foundation-model baseline.

## 2026-07-12 — Cached TESS unlabeled representation source v1

**Repo:** 2026 Exoplanet Research
**Data:** 11,960 existing cached TESS SPOC light-curve products across 2,790 TICs
and 84 represented sectors; 29.79762048 GB already on disk.
**Role:** Training only, for self-supervised objectives with no labels.
**Acquisition:** Metadata-only inventory; zero downloads and zero FITS reads.
**Decision:** Register the exact cache-relative product paths and
`mast:TESS/product/...` URIs after excluding every TIC in local labeled corpora
and the frozen `tess_live_search_v1` role.
**Leakage controls:** Group by TIC; forbid validation, calibration, frozen
evaluation, candidate discovery, and supervised label use. "Unlabeled" means
absent from the project's listed local labeled corpora, not astrophysically
label-free.
**Storage:** The committed row inventory is 6.1 MB. Raw products already occupy
29.79762048 GB inside the existing 37 GB TESS cache; no derived arrays are
authorized yet while the shared archive cache remains above 80 GB.
**Evidence:** Row SHA-256 `38a869666269367480a1aa9d2a29d76a86fb2ec6d2259a23635ad44f257c155a`;
summary SHA-256 `2a1b876f8983d9493a6dd4831406af623a754db6e024029793527d60234ce0e3`;
Run Report commit `2016811`.
**Next action:** Design a streaming or bounded-sample preprocessing benchmark
that estimates derived size and throughput before creating any training arrays.

## 2026-07-12 — Kepler masked-representation pilot v1

**Repo:** 2026 Exoplanet Research
**Data:** Existing 15,649-row `t1_1_kepler_master_combined` corpus
**Roles:** Existing predefined training, validation, and frozen-evaluation roles
**Acquisition:** No download; reuse the validated local corpus and its KIC-grouped splits.
**Decision:** Permit masked-reconstruction pretraining on training rows with labels
hidden from the objective, then freeze the encoder and fit a linear probe on
training labels. Validation may select the pretraining/probe states. The frozen
test split may be opened once for the versioned v1 result.
**Comparison:** Report grouped test AUC and top-100 positive yield against a
period/duration/flux-summary linear baseline and the frozen
`benchmark_cnn_v1` AUC 0.957211.
**Leakage controls:** Fail on any `group_key` overlap; compute normalization
from training rows only; never tune from test results; record exact corpus,
config, and checkpoint hashes.
**Rejected interpretation:** These labeled snippets are being treated as
unlabeled inputs, but they are not a broad unlabeled Kepler/TESS corpus. A
pilot pass would not satisfy the full master-guide data requirement or authorize
model promotion.
**Storage:** No new raw data. One ignored small checkpoint plus one committed
result manifest and Run Report; safely below the 100 GB project ceiling.
**Next action:** Merge the benchmark tooling, run it from merged `main`, and
commit the measured outcome separately without changing the gate after seeing
the frozen-test result.
**Measured outcome:** Merged-code MPS run completed in 33.5 seconds. Embedding
test AUC 0.832630 did not beat `benchmark_cnn_v1` AUC 0.957211. The frozen test
is now consumed for pilot v1 and must not be used to tune or rerun this compact
architecture. Retain the 72% top-100 yield versus the tabular baseline's 6% as
motivation only for a materially different, newly versioned data/model plan.

## 2026-07-12 — K2 empirical candidate-score context v1

**Repo:** 2026 Exoplanet Research
**Data:** 588 completed rows from `t1_2_k2pandc_calibration`
**Role:** Calibration (descriptive reuse; role unchanged)
**Acquisition:** No download; reuse the committed completed-predictions artifact
and calibrated stacking weights.
**Decision:** Permit the exact full-ensemble scores to define an empirical CDF
and observed negative fraction at or above a candidate score. Preserve the
reference numerator/denominator and K2-domain limitations on every output.
**Fail-closed fields:** `calibrated_score=null`, `decision_threshold=null`, and
`threshold_version=no_decision_threshold_v1` because T1-2 optimized AUC rather
than probability calibration or threshold utility.
**Rejected interpretation:** The observed tail-negative fraction is not a
guaranteed operational FDR bound. The catalog-selected K2 class balance is not
TESS live-search prevalence, and a zero observed fraction does not prove zero
false-discovery risk.
**Leakage:** No model fitting or threshold tuning occurs. The reference remains
the already-designated held-out calibration role and is not promoted to frozen
evaluation or training.
**Storage:** Negligible; one small sorted-score reference plus Run Report.
**Next action:** Merge the builder/output wiring, generate the reference from
merged code, validate hashes/counts, then commit the artifact separately.

## 2026-07-11 — TESS QLP live-search queue v1 execution

**Repo:** 2026 Exoplanet Research
**Data:** All 18 frozen `tess_live_search_v1` targets
**Role:** Live search
**Execution:** Three process shards, six workers each, TIC modulo split 7/4/7
**Outcome:** 18/18 targets processed in 72.79 seconds observed wall time; 56
schema-v2 records (53 signal rows, three null rows), zero failures
**Null targets:** TIC 280792173, TIC 355704481, TIC 355704482
**Review queue:** Three signals on two targets have FPP < 0.15. TIC 201251996
has two extremely deep signals (31.7% and 58.4%) with weak XGBoost support and
is prioritized for false-positive rejection. TIC 355651994_s02 has P=97.1618 d,
depth=1.1546%, FPP=0.0663, and remains an unreviewed candidate signal.
**Provenance constraint:** All review-queue signals have QLP provenance score
0.5625, below `tfop_ready`; no external submission is authorized.
**Durable evidence:**
`artifacts/manifests/tess_live_search_v1_run_summary.json`, the three committed
Run Report ledgers, and ignored shard SQLite/log artifacts whose hashes are in
the committed summary.
**Next action:** Do not rerun the frozen batch. Apply the full false-positive
diagnostic path to the review queue, deepest/weakest-support signals first.

## 2026-07-11 — TESS QLP live-search queue v1

**Repo:** 2026 Exoplanet Research
**Data:** 18 product-backed TIC targets in the immutable
`tess_live_search_v1` queue (plus two no-product rejections retained in the
canonical queue)
**Role:** Live search
**Acquisition mode:** Metadata-only preparation now; future raw acquisition is
`stream_process_evict`
**Estimated download GB:** 0.04565088 GB across 103 exact QLP products
**Actual download GB:** 0 GB; this decision performed metadata queries only
**Free space before:** 340.564 GB decimal as recorded by the batch manifest
**Training priority score:** Not applicable; this dataset is forbidden for
training and calibration
**Live search priority score:** 19.332–21.0 for the executable snapshot; every
included target clears the normal >=18 policy gate
**Storage cost penalty:** 0 for every target (each is <=5 GB)
**Why this data:** The queue freezes a small, uniformly selected faint-star
novelty batch with verified QLP availability and exact MAST URIs, closing the
row-level live-search manifest prerequisite for reproducible candidate-ledger
wiring.
**Why not alternatives:** SPOC is target-limited and repeatedly produced
no-data historical scans for this frontier. Target-pixel files are larger and
are reserved for candidate follow-up. Training, calibration, TOI, CTOI,
confirmed-host, and 200 previously scanned targets are excluded.
**Why this acquisition mode:** Metadata-first preparation establishes exact
size and product scope without materializing raw FITS files. A later scan can
fetch one target at a time and evict redownloadable raw products.
**Eviction or pin rule:** Pin queues, manifests, ledger records, and unresolved
candidate evidence; evict redownloadable raw light curves after durable results.
**Leakage risks:** The live-search queue is role-isolated and explicitly
forbidden for model training/calibration. It carries unlabeled targets only.
**Manifest:** `metadata/dataset_manifests/tess_live_search_v1.json` and
`data_selection/batch_manifests/tess_live_search_v1.json`
**Expected scientific or model-hardening value:** Enables exact source IDs,
product URIs, regeneration scope, and auditable null/candidate outcomes for the
first post-model-production TESS live-search batch.

## 2026-07-10 — Versioned dataset-manifest contract

**Repo:** 2026 Exoplanet Research
**Data:** Existing committed Kepler KOI, Kepler DR24 TCE, and held-out K2
stacking-calibration row manifests
**Role:** Training and calibration, unchanged from the existing data-role
registry
**Training priority score:** Not applicable; no new dataset is acquired
**Live search priority score:** Not applicable; no live-search data is selected
**Why this data:** These are the production model's actual source manifests and
therefore the first artifacts that must satisfy the Astrometrics master guide's
stable dataset ID, provenance, checksum, role, label-source, and caveat
contract.
**Why not alternatives:** TESS and JWST do not yet have equivalent committed
row-level production manifests in this repository; inventing placeholder
artifacts would weaken provenance. The contract is mission-neutral and will be
required when those manifests are created.
**Leakage risks:** None added. Existing target-group splits and role
restrictions are preserved; the dataset manifests describe rather than alter
membership.
**Manifest:** `metadata/dataset_manifests/*.json`, validated against
`metadata/dataset_manifest.schema.json` and the Pydantic contract in
`src/exo_toolkit/dataset_manifest.py`.
**Expected scientific or model-hardening value:** Every canonical eval,
injection-recovery package, model run, and candidate ledger record can now cite
a stable dataset ID whose local artifact is checksum-verified and whose role
and limitations are explicit.

---

This log satisfies `docs/astrometrics_data_selection_policy.md` for production-relevant dataset choices. Keep entries concise, source-contract-based, and tied to committed manifests or run reports.

## 2026-07-08 Decision

Decision id: `t1-1-kepler-master-corpus-promotion-candidate`

Data role: `training`, `validation`, `calibration`, and `frozen_eval` via predefined train/val/test splits.

Data selected:
- Original KOI-based T1-1 Kepler snippets from `metadata/t1_1_kepler_training_manifest.jsonl`.
- DR24 TCE expansion snippets from `metadata/t1_1_kepler_dr24_expansion_manifest.jsonl`.
- Master combined corpus recorded locally as `data/processed/t1_1_kepler_master_combined.jsonl`.
- Validated split directory recorded locally as `data/t1_1_kepler_master_cnn_splits/`.

Why this data:
- Primary source tables were verified through NASA Exoplanet Archive TAP schema discovery before use.
- Labels are row-level Kepler KOI dispositions or DR24 TCE `av_training_set` labels with explicit label mappings.
- The split source is predefined by target/group, preserving leakage safety across examples from the same star.
- The master corpus materially improved the held-out CNN gate versus the prior passing Kepler checkpoint: raw test AUC 0.9572, calibrated F1 0.8347, Brier 0.0580, ECE 0.0142.

Rejected alternatives:
- Old C1-C19/C20-style TESS/combined retraining loops: rejected by measured gates and documented in `docs/PRODUCTION_READINESS.md`.
- `Q1_Q17_DR25_TCE` as a label expansion source: rejected because the relevant label columns were confirmed empty.
- Synthetic positives for this supervised training phase: rejected by `docs/exoplanet_exomoon_dataset_handoff.md`.
- Kaggle or mirror datasets: rejected while primary NASA/MAST sources are available.

Contamination controls:
- Do not use live-search candidate evidence as training data unless a future manifest demotes that data into a training role and removes it from blind-discovery claims.
- Do not mix exomoon residual/anomaly-ranking data into this supervised exoplanet classifier.
- Do not promote local checkpoint artifacts into `models/` without explicit human approval plus a promotion PR containing reproducibility metadata.

Storage and retention:
- Raw FITS products are treated as re-downloadable batch cache and are evicted after verified snippet extraction.
- Durable state is the committed manifests, run reports, local artifact ledger, source snapshots, and this decision log.
- The approved promotion package should include only selected production artifacts, not broad raw archives or scratch checkpoints.

Current status:
- `checkpoints/cnn_t1_1_kepler_master/best.pt` is the promotion candidate.
- Promotion is pending explicit human approval.

## 2026-07-10 Decision

Decision id: `t1-2-k2pandc-held-out-calibration-set`

Data role: `calibration` (T1-2 stacking Tier-3 weight calibration, not model training).

Data selected:
- 596 rows (356 CONFIRMED / 240 FALSE POSITIVE) from the NASA Exoplanet
  Archive K2 planets-and-candidates table (`k2pandc`), recorded in
  `metadata/t1_2_k2_calibration_manifest.jsonl`.
- Bayesian + XGBoost scores computed catalog-only in
  `metadata/t1_2_k2_calibration_partial_predictions.jsonl`.
- Native K2 light-curve snippets pending a bounded live fetch to
  `data/t1_2_k2_calibration_snippets.jsonl` (local, human-run).

Why this data:
- T1-2 requires a held-out set never used to train `xgboost_koi.json` or
  promote `benchmark_cnn_v1`. Both were trained/split entirely from Kepler
  prime-mission KIC targets, and `data_role_registry.yaml`'s existing
  training/validation/frozen_eval entries for that corpus explicitly forbid
  reuse for stacking calibration.
- K2's own planet catalog uses a disjoint EPIC target catalog observed
  during different campaigns/sky fields than the original Kepler mission,
  so `k2:epic:<id>` group keys can never collide with the `kepler:kic:<id>`
  namespace used by T1-1 — zero leakage risk by construction, not by
  post-hoc deduplication.
- `k2pandc` was already identified in the Label-Source Discovery Protocol
  (`CLAUDE.md`) as the one untried lever for Kepler-family data (native K2
  light curves, as opposed to the already-fetched TESS-re-observed subset).
  Using it for calibration rather than training expansion is recorded here
  so a future agent does not also try to consume it for T1-1 training.
- Live-verified via `tap_schema.columns` before use: 2,261 usable rows
  (CONFIRMED/FALSE POSITIVE with valid period+epoch) as of 2026-07-10,
  comfortably above the ~500-example gate noted across
  `docs/PRODUCTION_READINESS.md`/`CLAUDE.md`.

Rejected alternatives:
- Reusing the Kepler master corpus's own validation or frozen-eval splits:
  explicitly forbidden by `data_role_registry.yaml` (`forbidden_uses`
  already lists "stacking weight calibration" / "full-ensemble stacking
  calibration after CNN promotion" for those roles).
- TESS-domain calibration data: rejected because `benchmark_cnn_v1` is a
  Kepler-domain-only checkpoint; pairing it with a TESS calibration set
  would measure a domain mismatch, not real production performance.
- The already-fetched `data/tess_k2_overlap_snippets.jsonl` (K2 ephemerides,
  TESS light curves): rejected — that corpus is TESS-domain flux, not
  native K2 flux, and (see bug note below) was built with a corrupted
  epoch and needs re-fetching regardless.

Contamination controls:
- Do not reuse these 596 EPIC targets for any future T1-1-style Kepler/K2
  CNN training-corpus expansion — recorded as a `forbidden_uses` entry in
  `data_role_registry.yaml`.
- Do not report T1-2 calibration AUC/weights as a benchmark_cnn_v1
  performance claim; that claim belongs solely to the frozen-eval split.

Bugs found and fixed while building this dataset (see `AGENTS.md` /
`docs/PRODUCTION_READINESS.md` for full detail):
1. `pl_tranmid` in `k2pandc` is already full BJD_TDB, not BKJD. An existing
   script, `Skills/fetch_tess_k2_overlap_snippets.py`, assumed BKJD and
   added the 2454833-day offset a second time, corrupting the phase-fold
   epoch for every one of its already-fetched 2,086 TESS-domain K2-overlap
   snippets (`data/tess_k2_overlap_snippets.jsonl`, local-only). Fixed in
   both that script and the new T1-2 manifest builder. No promoted model
   was trained on the corrupted data (it predates any C20-style TESS
   combined-corpus assembly), but the local file must be re-fetched before
   any future use.
2. `pl_trandur` is empirically already in hours, not days as
   `tap_schema.columns` itself claims ("Transit Duration [day]", unit=
   "day") — cross-checked against literature-published transit durations
   for K2-18 b and K2-3 b. Fixed before any conversion was applied
   downstream.

Storage and retention:
- k2pandc catalog rows and derived manifests/predictions are small,
  committed metadata — no large raw archive to bound.
- Native K2 raw FITS products (once fetched) follow the same
  re-downloadable-batch-cache policy as other mission fetchers; only the
  phase-folded snippet JSONL is retained locally.

Current status:
- **COMPLETE (2026-07-10)** — manifest, catalog-only scoring, live snippet
  fetch (588/596), CNN merge, and stacking calibration all done; calibrated
  weights are wired into production. See the 2026-07-10 "CNN merge and
  stacking calibration" entry below for the cross-mission decision record.

## 2026-07-10 Decision: T1-2 CNN Merge and Stacking Calibration (Cross-Mission Use)

Decision: Score the 588 fetched K2 calibration snippets with the Kepler-trained
`benchmark_cnn_v1` checkpoint (via `Skills/merge_t1_2_k2_cnn_predictions.py`),
explicitly declaring and logging this as a cross-mission application, then
apply the resulting calibrated stacking weights globally in
`cli.py`'s `full-ensemble` blend and `StackingScorer`'s defaults.

Rationale:
- `benchmark_cnn_v1` is trained exclusively on Kepler prime-mission targets.
  Scoring K2 targets with it is a real cross-mission application of the
  same kind the cross-mission scoring guard (PR #193, merged the same day)
  was built to catch — but `CnnScorer.from_checkpoint()` (used here) is not
  subject to that guard, which only gates `run_pipeline()`/`exo scan`.
- This is not a misuse of the guard's intent: T1-2's whole purpose is to
  *measure* real held-out CNN performance so the stacking calibrator can
  down-weight it appropriately if it does not transfer — exactly the
  "deliberate out-of-domain testing" case the guard's own
  `allow_cross_mission_cnn` escape hatch exists for. K2 was chosen for this
  calibration set specifically because it is leakage-safe from Kepler
  training data (see the 2026-07-10 Decision above) — not because it is
  the CNN's native domain.
- The measurement (after the sys.path bug below was fixed) found an
  AUC-maximising grid search over 588 held-out examples gives the CNN an
  optimal weight of **0.00** (XGBoost=0.95, Bayesian=0.05, best AUC 0.9576).
  This is **not** evidence the CNN carries no cross-mission signal -- its
  standalone AUC on this same held-out set is 0.7458, well above chance,
  consistent with its native-Kepler-domain frozen-eval AUC of 0.9572. The
  zero weight instead reflects that XGBoost alone already achieves 0.9575
  AUC on this catalog-derived set -- essentially the same as the full
  blend's 0.9576 -- so a coarse (0.05-step) pure-AUC grid search has no
  incentive to dilute an already-near-ceiling tabular classifier with a
  weaker (if real) flux-based signal. Whether CNN-derived features would
  earn nonzero weight under a different objective (e.g. one that also
  rewards calibration or robustness, not just AUC) is untested.
- Applying these weights **globally** (not conditioned on
  `cnn_scorer.training_mission == mission`) was a deliberate choice, made
  after explicit consideration of the alternative (keep the old 0.35/0.35
  weights for genuine same-domain Kepler scans). The old weights were never
  themselves calibrated -- they were an uncalibrated guess from
  `CNN_SPEC.md` -- so preserving them for the untested same-domain case is
  not "more correct," just differently untested. The cross-mission guard
  already blocks the CNN by default for any mismatched mission, so the only
  scenario where the global constant applies without an explicit
  `--allow-cross-mission-cnn` override is genuine same-domain Kepler
  scoring -- a narrower, secondary use case (TESS is this project's primary
  discovery target) for which conservative defaults are preferred per
  `CLAUDE.md`'s Scientific Guardrails.

Rejected alternative:
- Mission-conditional blending (new calibrated weights only when
  cross-mission, old 0.35/0.35 preserved for matched-mission Kepler scans):
  rejected because it adds real complexity to a scoring hot path to protect
  a case with zero calibration evidence backing either number, and because
  a genuine same-domain stacking calibration would require its own
  dedicated held-out Kepler set (distinct from the existing
  training/validation/frozen-eval roles) -- out of scope unless explicitly
  requested as future work.

Bug found and fixed in the process: `calibrate_stacking_weights.py`'s
`load_predictions_jsonl()` called `float(rec["cnn_prob"])` unconditionally,
which would raise `TypeError` on any row with a null `cnn_prob` (e.g. an
EPIC target whose snippet fetch failed). Fixed with a defensive skip in the
loader itself, plus `merge_t1_2_k2_cnn_predictions.py`'s own
`filter_complete_rows()` as defense-in-depth so the on-disk completed-
predictions artifact never contains a null `cnn_prob` row in the first
place.

SEVERE bug found and fixed in the process: the first live run of
`merge_t1_2_k2_cnn_predictions.py` produced `cnn_prob=0.5` for all 588 rows
-- a flat constant, not real predictions -- with zero error or warning.
Root cause: `CnnScorer._ensure_model()` does the absolute import
`import_module("Skills.cnn_inference_batcher")`, which needs the repo root
(not just `src/`) on `sys.path`. Running a `Skills/*.py` script directly
(`python Skills/foo.py`, this project's standard invocation pattern) only
gets the script's own containing directory auto-prepended by Python, never
the repo root, so the import silently raised `ModuleNotFoundError`, caught
by a bare `except Exception`. Every prediction thereafter returned the
neutral fallback with no indication anything was wrong -- and the resulting
bogus `calibrate_stacking_weights.py` run happened to land on the exact
same `CNN=0.000` weight as the corrected re-run, which is precisely what
made this dangerous: a coincidentally-plausible wrong answer, only caught
by manually checking that `cnn_prob` had exactly one distinct value across
all 588 rows. Fixed with a new `_ensure_repo_root_on_sys_path()` helper in
`src/exo_toolkit/ml/cnn_scorer.py` (idempotent, called before every
`Skills.` import site) plus a `RuntimeWarning` on any genuine load failure
that is not the documented "PyTorch not installed" case. 9 new regression
tests, including one that reproduces the exact incident. See
`docs/PRODUCTION_READINESS.md` T1-2 for the full record.

Future work (not requested, recorded for a future agent): a genuine
same-domain stacking calibration for Kepler-mission `full-ensemble` scoring
would need a fresh, dedicated held-out Kepler set not already claimed by
`data_role_registry.yaml`'s training/validation/frozen-eval roles.

## 2026-07-19 Decision: Hunter New/Follow-Up Search Lifecycle

- Date: 2026-07-19
- Repo: 2026 Exoplanet Research
- Data: live TIC/MAST metadata candidate universe and durable follow-up registry
- Role: `live_search` or `followup_live_search`, selected explicitly by mode
- Acquisition mode: metadata-only selection, then target-batch light-curve pull
- Estimated download GB: per-search sum frozen in candidate snapshots when known
- Actual download GB: recorded by exact raw-product provenance during execution
- Free space before: checked by the existing fetch/cache policies at execution
- Free space after: recorded by the applicable acquisition/run evidence
- Training priority score: not applicable; live targets remain training-forbidden
- Live search priority score: deterministic TIC suitability plus verified data
  availability and storage penalty; evidence-based follow-up priority for prior
  candidates
- Storage cost penalty: 0/1/2/3/5 at <=5/25/100/250/>250 GB
- Why this data: it closes the required production path from a large target
  universe to exact durable searches, results, and recommended follow-ups.
- Why not alternatives: the seven-target background runner is fixture-only;
  dynamic `star_scanner.py` execution does not preserve a general pending-search
  lifecycle and must not silently regenerate membership.
- Why this acquisition mode: metadata-first ranking evaluates the broad pool
  without raw downloads; only the immutable selected manifest enters the
  acquisition pipeline.
- Eviction or pin rule: SQLite state and Run Reports are durable; raw public
  light curves remain re-downloadable cache unless candidate-evidence policy
  pins them.
- Leakage risks: both live roles forbid training, validation, calibration, and
  frozen-eval reuse without a separately versioned role transition.
- Manifest: `search_manifests`/`search_manifest_targets` in
  `data/hunter_searches.sqlite3`, SHA-256 stamped; CSV is review-only.
- Search category: `new_target_search` for `--mode new`, `followup_search` for
  `--mode follow-up`.
- Expected scientific or model-hardening value: exact reproducibility,
  no-overwrite history, loud partial failure, safe resume, and actionable
  follow-up continuity across sessions.
- Citations: `docs/astrometrics_data_selection_policy.md` and
  `docs/HUNTER_PRODUCTION_WORKFLOW.md`.

## 2026-07-12 Decision: Canonical Regression Evaluation v1

- Date: 2026-07-12
- Repo: 2026 Exoplanet Research
- Data: `canonical_real_controls_v1` and `canonical_injected_controls_v1`
- Role: `frozen_eval` (separate real-only and synthetic-inclusive roles)
- Acquisition mode: metadata-only reuse plus deterministic local generation
- Estimated download GB: 0
- Actual download GB: 0
- Free space before: 307 GiB
- Free space after: unchanged within filesystem reporting precision
- Training priority score: not applicable
- Live search priority score: not applicable
- Storage cost penalty: negligible (4.7 KB config plus baseline report)
- Why this data: closes the Phase 1 canonical-regression-eval gap with one
  accepted confirmed-planet control, one accepted catalog false-positive
  control, and two deterministic injected-transit controls.
- Why not alternatives: no live archive fetch is required; the accepted formal
  suite is already durable evidence, while the injected controls exercise the
  current search implementation offline.
- Why this acquisition mode: the shared Lightkurve cache is above the 80 GB
  caution threshold, and this regression gate needs no new raw data.
- Eviction or pin rule: commit and retain the small immutable baseline,
  configuration, manifests, and registry entries indefinitely.
- Leakage risks: both roles forbid training, calibration, and threshold tuning;
  synthetic controls are explicitly separated from the real-only role.
- Manifest: `metadata/dataset_manifests/canonical_real_controls_v1.json` and
  `metadata/dataset_manifests/canonical_injected_controls_v1.json`
- Expected scientific or model-hardening value: every pipeline change can now
  report sample-level outcome and metric deltas against a versioned baseline.
- Citations: `docs/astrometrics_coding_agents_master_guide.md` Phase 1.5 and
  `artifacts/manifests/formal_acceptance_v2.json`.
