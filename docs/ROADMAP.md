# ROADMAP

This roadmap is the executable, repo-specific projection of
`docs/astrometrics_coding_agents_master_guide.md`. The master guide controls
strategic phase ordering; this file records what this repository has actually
implemented and the remaining production work. A utility's existence does not
by itself satisfy a master-guide evidence requirement.

## Current Master-Guide Alignment — Production Priority Order

- **Adversarial Hunter requalification (0.5.3; delivery gate in progress):**
  Version 0.5.2 is not PROD under the stricter closure contract. Remove all
  production sibling-repository filesystem reads, constrain cross-project
  history to verified copies inside this repository, and add one deterministic
  fresh-datastore acceptance through the installed `EXO-Hunter` shell and
  canonical new/follow-up pipeline. The harness must replace only external
  network boundaries, exercise the real scorers and durable lifecycle, prove
  adaptive outside-partition discovery, invalid/stale/history/alias handling,
  best-available weak candidates, exact execution, partial-failure resume, and
  restart state, then emit a hash-verifiable evidence bundle and datastore
  snapshot. The implementation and v16 clean-state acceptance now pass at
  `79f606e` with 17/17 assertions and an independently verifiable schema-v6
  snapshot. Full local gates, PR CI, squash merge, synchronization, and
  merged-state verification remain required before restoring PROD status.

- **Explainable operator reporting (0.5.2 PROD accepted):**
  The candidate grid and timestamped CSV now expose the full decision context
  required to review a selection: stable identities, classification, distance,
  storage, explicit search status, prior-search count/provenance, rank,
  selection reason, and exoplanet metrics. This is a presentation/export delta
  over the unchanged accepted 0.5.1 lifecycle. Clean implementation and
  merged-main gates, PR #326 CI, squash merge `a062245`, synchronization, and
  v15 structured acceptance all passed under the prior contract; that status
  is historical and superseded by the 0.5.3 requalification item above.

- **Exact shared Hunter command contract (0.5.1 PROD accepted):**
  Add the named `EXO-Hunter` launch command and canonical
  `/Create-New-Search --targets N --mode ...` and `/Run-New-Search` slash
  commands as aliases over the accepted 0.5.0 terminal and lifecycle. Clean
  10/10 gates with 3,214 tests, all PR #324 CI events, squash merge `12ec767`,
  merged-main CI, installed probes, and exact Git-tree identity pass. Evidence
  is `artifacts/manifests/hunter_live_acceptance_v14.json`.

- **Persistent Hunter terminal (0.5.0 PROD accepted):**
  The current business contract requires the shell-launched `ExoHunter`
  application, persistent slash discovery/history/completion, direct canonical
  new/follow-up execution, domain-specific real-work animation, and safe
  non-interactive operation. Implementation and real new/follow-up business
  workflows pass on commit `a854dd4`; PR #322 passed all CI runs and
  squash-merged as `6b8c78e`; clean synchronized merged-main quality is 10/10
  with 3,212 tests. Current evidence is
  `artifacts/manifests/hunter_live_acceptance_v13.json`.

- **Hunter production lifecycle (0.4.0 PROD accepted):**
  The 2026-07-25 adversarial audit supersedes the older PROD label. The active
  closure plan is the seven-gate plan at the top of
  `docs/HUNTER_PRODUCTION_WORKFLOW.md`: broad catalog paging with top-N
  sufficiency, no operator candidate bypass, atomic pre-write validation,
  explicit decision validity, durable cross-project identity history,
  reconciled contracts, and current end-to-end/quality/CI evidence. No human
  blocker is open. All implementation, live business-acceptance, local quality,
  PR CI, squash-merge, and merged-main synchronization gates pass. The current
  acceptance is `artifacts/manifests/hunter_live_acceptance_v12.json`; the
  historical narrative below remains evidence of prior increments, not
  authority for the current completion claim.

- **Historical Hunter production lifecycle increments:**
  version 0.3.2 adds the missing durable bridge from a large candidate universe
  through exact pending-search execution and evidence-based follow-ups. The
  append-only SQLite contract separates candidate catalog, immutable manifest,
  run attempts, target history, and follow-up registry; selection is
  deterministic and non-LLM, partial runs are loud, and restart retries only
  unfinished/failed targets. `Create-New-Search`, `Run-New-Search`, and
  `Show-Follow-Ups` are packaged shell entry points. Offline end-to-end and
  100-from-10,000 tests pass; 0.3.1 fixes the packaged-process repository-Skill
  loader exposed by the first merged-main live attempt. The next run completed
  real selection/execution but exposed nested scorer fields that 0.3.2 now
  consumes for composite selection and follow-up gating. The post-fix
  provenance-linked follow-up run completed on merged main with the correct
  composite, zero failures, intact history, valid SQLite/foreign keys, and a
  pushed Run Report. The original acceptance incorrectly treated test-only
  follow-up creation as production evidence while the live registry was empty;
  its unchanged historical artifact is superseded by
  `artifacts/manifests/hunter_live_acceptance_v1_reassessment.json`. Version
  0.3.3 supplies checksum-verified reviewed-result import and separates visible
  recommendations from immediately executable searches. Merged-main import now
  durably records the real reviewed TIC 355651994 evidence; display,
  recommendation, selection exclusion, idempotency, SQLite integrity, foreign
  keys, and Run Reports all pass. Version 0.3.4 makes report-push failure loud.
  The v2 acceptance is preserved, but a 2026-07-21 audit found that actionable
  rows remained open after scheduling/completion and could be selected again.
  Version 0.3.5 adds append-only lifecycle events, consuming-search links,
  failure-safe resume semantics, terminal disposition, parent-child
  recommendations, and stale-candidate rejection. The first real migration was
  integrity-clean and exposed one remaining mismatch: a non-executable row was
  labeled `open`. Version 0.3.6/schema v4 corrects old rows append-only and
  creates new waiting-for-data recommendations as `deferred`. Both corrective
  PRs are merged; production schema v4 is integrity-clean and the row has
  append-only `open → deferred` history. A current metadata-only MAST check
  still finds only sectors 1 and 28, neither event-covering, so the real row is
  correctly deferred. That is an expected scientific state, not a product
  blocker. The next audit found the actual gap: default follow-up creation reads
  only open registry rows while seven preserved discovery logs contain 608
  events across 200 targets. Version 0.3.7 adds a committed normalized history
  contract, idempotent durable import, exact preservation of failures and
  repeated work, and deterministic full-universe eligibility/ranking.
  Merged-main import and repeat-import evidence now pass: 608 source events are
  durable, 202 targets are evaluated with explicit dispositions, integrity and
  foreign keys are clean, and no empty-registry inference remains. Future
  observations may reopen the deferred row but are unrelated to production
  readiness. See
  `artifacts/manifests/hunter_live_acceptance_v3.json`,
  `artifacts/manifests/hunter_live_acceptance_v2_reassessment.json`, and
  `docs/HUNTER_PRODUCTION_WORKFLOW.md`. Replacement acceptance is
  `artifacts/manifests/hunter_live_acceptance_v4.json`. Version 0.3.8 closes
  the subsequent validity/provenance audit with byte-verified source imports,
  typed execution/failure and model-artifact identity, schema-v5 storage
  immutability, full content/relationship verification, explicit information-
  gain ranking inputs, and evidence-derived acceptance tests. The recomputed
  schema-v5 snapshot and structured requirement evidence are recorded in
  `artifacts/manifests/hunter_live_acceptance_v5.json`. A post-merge loop found
  that the changed ranking still used an older generic selector ID. Version
  0.3.9 closes that provenance gap with distinct new/follow-up/operator
  selector versions and verifier-enforced frozen formulas, weights, thresholds,
  information-gain definitions, and candidate/manifest identity agreement.
  Current acceptance is `artifacts/manifests/hunter_live_acceptance_v6.json`.

1. **Phase 1 — validated manifest contract (CONTRACT COMPLETE):** version 0.2.30 adds
   the shared schema, frozen Pydantic contract, fail-closed checksum/path
   validator, stable dataset IDs, and data-role-registry links for the committed
   Kepler/K2 production datasets. Require the same contract when the first TESS
   or JWST row-level production manifest is created; do not invent placeholder
   datasets merely to claim mission coverage.
2. **Phase 1 — reproducible candidate ledger (LIVE EVIDENCE + FIRST REVIEW COMPLETE):** versions 0.2.31, 0.2.34, 0.2.36, and 0.2.37
   add a strict, mission-neutral, append-only provenance contract and SQLite
   table carrying the master-guide fields (`source_dataset_id`, raw URI,
   preprocessing/generator versions and parameters, model versions/scores,
   injection context, review state, and regeneration command). Next, wire every
   production scan path to write this contract rather than relying on the
   legacy TESS-only convenience table. Version 0.2.34 corrects the contract
   for stitched light curves by preserving every exact archive product URI.
   Version 0.2.36 supplies the first real row-level live-search source:
   `tess_live_search_v1`, an immutable 18-target queue with 103 exact QLP
   product URIs and a checksum-validated dataset manifest. Version 0.2.37 wires
   the frozen queue to schema-v2 SQLite writes for candidates, scientifically
   useful null results, and preprocessing failures. Execution fails closed if
   the dataset checksum, target membership, product inventory, or exact fetched
   URI tuple differs; process shards receive collision-free log, database, and
   run-report paths. The three-shard live execution completed successfully on
   2026-07-11: 18/18 targets, 56 schema-v2 rows, three high-priority nulls, and
   zero failures. The committed run summary preserves local-ledger/log hashes
   and places three low-FPP signals on a conservative review queue. The first
   review is complete: two eclipse-scale signals are `likely_false_positive`;
   the remaining signal is `plausible_but_weak` because limb-darkening fails
   and centroid/odd-even evidence is missing. Version 0.2.39 adds the bounded,
   Run-Report-enabled TESS-SPOC target-pixel centroid diagnostic needed for
   that review. Its live run established that neither available TPF sector
   covers a predicted 97.16-day event; version 0.2.40 persists that no-coverage
   result rather than raising without evidence. Odd/even is likewise
   observationally underdetermined: only two independent QLP events exist and
   the diagnostic requires four. The requested follow-up is complete to the
   limit of available observations; do not add more ledger scaffolding or run
   another blind scan.
3. **Phase 1 — canonical regression evals (COMPLETE):** version 0.2.41 adds
   an offline sample-level suite with separate real-only and synthetic-inclusive
   frozen-evaluation roles. It preserves the accepted pi Mensae c and TOI-146.01
   controls, reruns deterministic deep/subthreshold injected-transit controls,
   and reports per-case deltas against the committed v1 baseline.
4. **Phase 2 — production sensitivity evidence (BOUNDED V1 + EXPANDED V2
   COMPLETE):** generic BLS transit
   injection-recovery tooling exists, and version 0.2.42 adds a bounded,
   cache-only real-background runner tied to the production pipeline,
   frozen-eval dataset ID, and `benchmark_cnn_v1` promotion package. A
   36-trial pre-merge smoke completed with zero failures in 7.1 seconds using
   six workers. The first merged run was failed closed because quarter-filtered
   inputs retained all-quarter provenance URIs; version 0.2.43 fixes and tests
   aligned path/curve filtering. The corrected merged-code run is now committed
   at `artifacts/manifests/production_sensitivity_v1.json`: 23/36 recoveries,
   zero failures, and curves by period, depth, duration, and real-background
   label. This closes the bounded short-period evidence deliverable, not general
   completeness. The v1 depth bins are non-monotonic under a 12-sample/bin,
   strongest-peak-only search and must not be interpreted as a survey
   completeness law.
   Version 0.2.46 commits the merged-code v2 artifact: 8/16 recovered in 7.98
   seconds with zero failures. Moderate TTV and both gap scenarios recovered;
   strong TTV, 90-day periodic, single-transit, and low-depth variability cases
   did not. These are measured boundary cases, not favorable assumptions.
5. **Phase 2 — calibrated candidate context (COMPLETE):** extend production
   outputs with calibration dataset IDs, score quantiles, threshold versions,
   and empirical false-discovery context now that canonical eval and sensitivity
   sets exist. Version 0.2.47 adds a validated reference contract, builder, CLI
   output wiring, and schema-v2 ledger preservation for the exact calibrated
   full-ensemble score. It deliberately emits null probability calibration and
   null decision threshold while reporting empirical K2 score rank and observed
   tail-negative fraction with numerator/denominator and domain limitations.
   Version 0.2.48 commits `models/candidate_context_v1.json`, built from merged
   code over all 588 reference rows with exact source hashes and a successful
   Run Report. No probability calibration or threshold is claimed.
6. **Phase 3 representation benchmark (BOUNDED PILOT COMPLETE; CNN GATE NOT
   MET):** version 0.2.49 adds a bounded masked-reconstruction Transformer
   pilot over the existing predefined KIC-grouped Kepler splits. Labels are
   hidden during training-only pretraining; the encoder is frozen before a
   linear probe; validation selects states; and the frozen test split opens
   once against a period/duration/flux-summary linear baseline and
   `benchmark_cnn_v1`. Version 0.2.50 commits the merged-code MPS result:
   embedding test AUC 0.832630/F1 0.635135/top-100 yield 72%; tabular test AUC
   0.823495/F1 0.274336/top-100 yield 6%; CNN AUC 0.957211/F1 0.834688. The
   compact pilot is rejected as a CNN replacement and must not be repeated
   unchanged. Phase 3 remains active for a materially different experiment
   backed by broad unlabeled Kepler/TESS data, stellar-variability labels,
   injection-recovery comparison, and external foundation-model baselines.
   Version 0.2.51 adds the cache-only source-inventory builder needed before any
   follow-up: the 41 GB Kepler cache contains only 11 KICs outside the labeled
   master corpus, while the 37 GB TESS cache contains 2,797 TICs outside local
   labeled corpora before live-role exclusion. No new archive download is
   authorized while the shared cache remains above the 80 GB caution threshold.
   Version 0.2.52 commits the merged-code result: 11,960 exact SPOC products
   across 2,790 eligible TICs and 84 sectors,
   totaling 29.79762048 GB already cached. The dataset is registered training-
   only. Next, benchmark streaming/bounded-sample preprocessing size and
   throughput before authorizing any derived arrays or new model training.
   Version 0.2.53 removes the six-terminal operational burden for future
   shard-capable acquisition: one fail-closed supervisor now launches the
   measured six shards × six workers, isolates logs, enforces storage/repo
   preflight, propagates shard failures, and serializes shard Run Report git
   operations. It does not authorize a new download or change the next Phase 3
   evidence gate. The same release replaces sequential local validation with a
   single-parent quality runner: six disjoint pytest file shards × six xdist
   workers execute beside Ruff and mypy, with per-gate logs and a combined
   failure status. Its optimized run passed 2,718 tests plus both static gates
   in 34.1s, about 58% faster than the earlier 81.57s xdist baseline.
   This is an operability improvement, not new scientific evidence.
   Version 0.2.54 adds the bounded preprocessing benchmark that the inventory
   gate requires: one parent supervises six Python shard subprocesses with six
   FITS workers each, opens a deterministic 36-product/36-TIC sample spanning
   sectors 1-98, filters `QUALITY == 0`, robustly normalizes and resamples to
   2,048 float32 bins in memory, then discards every derived array. It verifies
   the committed training-only inventory hash, paths, and file sizes, downloads
   nothing, and projects full-corpus time/size from measured results. Tooling
   and a 6/6 real-cache process smoke are complete. The merged-code 36-product
   run then passed 36/36 with zero failures/downloads/persisted arrays at 85.77
   products/s. It projects the full 11,960-product normalized-flux-only
   transform at 97.98 MB and 139.44 seconds, so future work should stream it
   rather than create a durable derived corpus. This closes the preprocessing
   measurement gate, not Phase 3: stellar-variability labels,
   injection-recovery comparison, and an external foundation-model baseline
   remain required before a materially broader representation experiment can
   support any production-model decision. The release gate passed 2,726 tests
   plus Ruff/mypy in 34.1 seconds.
   Version 0.2.55 freezes the external-baseline source contract before any
   dependency or weight download: Chronos-Bolt tiny and Astromer2 provide the
   bounded general time-series and astronomy-native controls at exact commits,
   hashes, and sizes. Three Python 3.14-compatible direct wheels plus both ONNX
   files total 56,036,648 bytes. The metadata-only verifier fails closed and
   downloads zero payload bytes. Next, run that verifier from merged `main` and
   commit its evidence artifact; only then may a separate bounded inference
   smoke consider optional dependency installation. Scientific Phase 3 still
   requires variability labels and injection-recovery comparison. The local
   release gate passed 2,733 tests plus Ruff/mypy in 31.1 seconds.
   The first merged verifier run failed closed because the default Python
   opener followed Hugging Face's 302 into Xet and lost the resolver's pinned
   commit/size/hash headers. Version 0.2.56 preserves the authoritative initial
   HEAD response, covers it with an offline regression test, and passes a live
   read-only header smoke. Its 6×6 gate passed 2,734 tests plus Ruff/mypy in
   26.1 seconds. The merged full verifier subsequently passed 7/7 operations in
   4.94 seconds, verifying all five pinned sources and 56,036,648 projected
   direct bytes with zero payload downloads (artifact SHA `5610bbb8…3042`, Run
   Report commit `ae4e659`). Source identity/footprint is complete. Next design
   a bounded inference smoke that measures ONNX memory/numerics before any
   optional dependency or weight installation; the scientific experiment also
   still requires variability labels and injection-recovery comparison.
   Version 0.2.57 implements that smoke: one deterministic cached SPOC product,
   at most 2,048 cadences, two exact-revision models, isolated one-thread CPU
   sessions, finite 256-dimensional output checks, and structured timing/RSS
   evidence. The optional dependency group and ignored model-cache ledger keep
   the default runtime and `git add .` safe. The next planned step was a merged
   pip dry-run, optional-group install, and smoke execution. A pass would close
   runtime integration only, not the variability/injection/scientific gates.
   The local 6×6 gate passed 2,743 default tests plus Ruff/mypy in 24.2 seconds
   with optional inference packages absent.
   The first merged smoke then failed closed before payload download because
   Xet attempted a sandbox-blocked home-cache log. Version 0.2.58 relocates
   `HF_HOME` and `HF_XET_CACHE` into the ignored model cache before import; the
   optional group is installed, `pip check` passes, and the 6×6 gate passed the
   same 2,743 tests plus Ruff/mypy in 26.2 seconds. The merged retry passed both
   exact models in 26.875 seconds with finite `(1,1,1,256)` outputs and max
   child RSS 186,204,160 bytes. Artifact SHA is `1cc59ab3…5de5d10`; Run Report
   commit `f8a7207`. Version 0.2.59 closes runtime integration. Version 0.2.60
   pins the 47,055-row Drake et al. Catalina variable-star catalog and adds a
   fail-closed, zero-full-payload metadata verifier. The merged verifier passed
   5/5 operations in 3.334 seconds; artifact SHA is `eb5d4bc6…39b9a` and Run
   Report commit is `b0003bb`. Version 0.2.61 records source identity complete.
   Next design and evidence the leakage-safe 2,790-TIC coordinate crossmatch
   using the measured single-parent six-shard/six-worker shape.
   Version 0.2.62 implements only a deterministic 216-TIC pilot under that
   shape, with exact-ID MAST batches, one hash-pinned shared Catalina cache,
   precommitted match safeguards, disjoint outputs, and training disabled. Run
   it from merged clean `main`, reconcile all six shards globally, and use the
   measured throughput/errors/overlap to decide whether a separate full-corpus
   contract is scientifically and operationally justified.
   Version 0.2.65 supersedes the failed MAST-column v1 contract with immutable
   v2 (`duplicate_id`, live verified) while preserving the same frozen pilot
   population and all scientific safeguards.
   Version 0.2.67 closes the pilot evidence gate: all 216 TICs queried cleanly,
   but none has a Catalina candidate within 3 arcseconds. Full execution and
   training are not authorized. The next roadmap item is a new immutable label
   source contract whose metadata-only preflight demonstrates materially
   better overlap with the frozen TESS inventory before any payload/training.
   Version 0.2.68 implements that item for ASAS-SN Catalog X: its immutable
   contract and exact-TIC preflight pin 378,861 rows and a zero-payload 6x6
   reproduction. The exploratory implementation-decision probe found 48/2,790
   exact matches, including 44 known variables at probability at least 0.902,
   with no duplicate TIC rows. Durable evidence remains pending the merged-main
   six-shard run and global reconciliation. Even a PASS authorizes only a
   separately reviewed benchmark design because the catalog classes are
   automated outputs, not human ground truth; training stays unauthorized.
   The version 0.2.68 release gate passed 2,772 default tests plus Ruff/mypy as
   8/8 supervised gates in 31.3 seconds under the canonical 6x6 topology.
   The merged preflight reproduced 48/2,790 unique exact-TIC matches in 6.762
   seconds observed wall time: 44 known variables, four discoveries, minimum
   probability 0.902, zero duplicate TIC/source IDs, and zero catalog payload
   bytes. Version 0.2.69 commits and integrity-tests every shard artifact and
   the passing aggregate. The source-overlap item is complete. Next design the
   bounded embedding-aware variability/injection benchmark around these 48
   training-disabled rows; do not train on the automated ASAS-SN classes.
   The version 0.2.69 evidence-release gate passed 2,773 default tests plus
   Ruff/mypy as 8/8 supervised gates in 32.3 seconds under the canonical 6x6
   topology.
   Version 0.2.70 completes that design gate with a hash-pinned, cache-only
   48-TIC contract and harness: four frozen injection cells produce 192 unique
   blind-BLS trials and 384 Chronos-Bolt tiny/Astromer2 model rows. Execute it
   only from merged clean `main` via the reviewed 6x6 supervisor and globally
   reconcile zero failures, duplicates, downloads, or persisted embeddings.
   The version 0.2.70 release gate passed 2,780 default tests plus Ruff/mypy as
   8/8 supervised gates in 33.3 seconds under the canonical 6x6 topology.
   Version 0.2.71 hardens the pending run by verifying every aggregate-owned
   ASAS-SN shard path/hash and rejecting incomplete, duplicate, or training-
   authorized source rows before FITS access. Its 6x6 gate passed 2,781 tests
   plus Ruff/mypy in 30.1 seconds.
   The merged run passed all seven global checks over 48 TICs, 192 trials, and
   384 model rows, with zero failures, duplicates, downloads, or persisted
   embeddings. Both models showed 96/96 higher-depth larger-shift comparisons;
   blind BLS recovered 13/192 trials. Version 0.2.72 commits and integrity-tests
   the evidence. This closes the bounded variability/injection gate but does
   not authorize training or a production change. The next Phase 3 item is a
   separately contracted grouped benchmark on labeled planet/false-positive
   data that retains the frozen CNN and classical baselines.
   The version 0.2.72 evidence-release gate passed 2,782 default tests plus
   Ruff/mypy as 8/8 supervised gates in 37.2 seconds under the canonical 6x6
   topology.
   Version 0.2.73 defines that next grouped gate over 1,536 unique cache-local
   Kepler KICs. It freezes balanced predefined train/validation/test subsets,
   exact cache and model identities, identical validation-selected linear
   probes, the frozen CNN and statistical comparators, and a one-time test
   opening. An external representation must beat the better required comparator
   by at least 0.01 on a precommitted ranking metric. Execute only from merged
   clean `main` through the reviewed cache-only 6x6 supervisor; training, broad
   extraction, promotion, and production changes remain unauthorized.
   The version 0.2.73 release gate passed 2,789 default tests plus Ruff/mypy as
   8/8 supervised gates in 34.1 seconds under the canonical 6x6 topology.
   The first merged v1 execution failed closed before processing because
   Kepler uses `SAP_QUALITY`, not v1's TESS-style `QUALITY`; zero benchmark data
   was downloaded or persisted. Version 0.2.74 preserves v1 and activates
   immutable v2 with the corrected schema plus an exact fingerprint of 111
   known truncated products. Only those paths may be skipped, every KIC must
   retain readable data, and every other FITS/schema error remains fatal.
   The version 0.2.74 release gate passed 2,790 default tests plus Ruff/mypy as
   8/8 supervised gates in 36.3 seconds under the canonical 6x6 topology.
   The first merged v2 run failed closed because its 95%-occupied 2,048-bin
   rule did not match all frozen KIC phase coverage; zero embeddings or durable
   outputs were written. Version 0.2.75 preserves v2 and activates immutable v3
   with neutral median physical-flux fill for empty bins. The full cache-only
   1,536-KIC preparation preflight and both-model smoke passed before release.
   The version 0.2.75 release gate passed 2,791 default tests plus Ruff/mypy as
   8/8 supervised gates in 36.3 seconds under the canonical 6x6 topology.
   The merged v3 run passed all 1,536 grouped KICs with 111 exact skips and zero
   failures/downloads/persisted embeddings. The CNN retained test AUC/AP/top-100
   0.923096/0.899184/91 versus Chronos-Bolt tiny 0.722778/0.696344/71 and
   Astromer2 0.708984/0.659679/67. Version 0.2.76 records the precommitted
   `no_external_added_value` outcome. Do not begin broad extraction or training
   with these models; this external-baseline branch of Phase 3 is closed.
   The version 0.2.76 evidence-release gate passed 2,797 default tests plus
   Ruff/mypy as 8/8 supervised gates in 36.3 seconds under the canonical 6x6
   topology.
7. **Phase 4 — individual anomalous-transit detector (BOUNDED CORE +
   DEPTH/ASYMMETRY/MISSING/EXTRA-EVENT EXTENSION COMPLETE):** version 0.2.77
   fixes the production vetter's hard-coded
   missing per-transit durations and midpoints. Each expected event now uses a
   local sideband baseline, a twice-noise half-depth gate, a flux-deficit-
   weighted midpoint, and cadence-resolved duration. At least two events must
   resolve; otherwise duration and timing diagnostics remain unavailable.
   This activates the existing duration-consistency and TTV feature paths,
   rejects flat noise, and recovers a tested 30-minute shifted event without
   adding data acquisition or model training. Future Phase 4 work may extend
   this bounded core to depth/asymmetry/missing/extra-event ranking, but must
   validate against real or frozen injected controls before production claims.
   The version 0.2.77 release gate passed 2,801 default tests plus Ruff/mypy as
   8/8 supervised gates in 35.3 seconds under the canonical 6x6 topology.
   Version 0.2.78 begins that named extension with `missing_transit_fraction`:
   the fraction of predicted transit windows with enough cadence coverage to
   test that never resolved a significant dip, reusing the exact resolution
   test already proven for duration/midpoint measurement. Wired as evidence
   against `planet_candidate` (−0.70) and for `instrumental_artifact` (+0.60);
   `None` unless at least two windows have coverage to test. See
   `docs/SCORING_MODEL.md §23`. Depth/asymmetry and extra-event ranking remain
   the next bounded increments of this roadmap item.
   The version 0.2.78 release gate passed 2,813 default tests plus Ruff/mypy as
   8/8 supervised gates in 28.2 seconds under the canonical 6x6 topology.
   Version 0.2.79 adds the second named increment, `transit_asymmetry`: for
   each event that already resolves a significant dip, the resolved-cadence
   deficit sum is split by sign of offset from the predicted center (not the
   resolved weighted midpoint) and recorded as a normalized before/after
   imbalance, reusing the exact resolved-cadence set already produced for
   duration/midpoint measurement. `transit_asymmetry_score()` is the RMS of
   these imbalances relative to a 0.30 threshold; wired against
   `planet_candidate` (−0.50) and for `instrumental_artifact` (+0.50); `None`
   unless at least two events resolve. See `docs/SCORING_MODEL.md §24`.
   Extra-event ranking remains the last named increment of this item.
   The version 0.2.79 release gate passed 2,828 default tests plus Ruff/mypy as
   8/8 supervised gates in 25.2 seconds under the canonical 6x6 topology.
   Version 0.2.81 closes the item with its third and final increment:
   `_measure_extra_events()` masks cadences near any predicted transit center
   across the full baseline, flags remaining out-of-transit cadences ≥3σ
   below the OOT median (MAD-based robust sigma), and clusters contiguous
   flags into events — a cluster counts only if it spans ≥2 cadences and
   ≤2× the transit duration. `extra_event_score()` is wired against
   `planet_candidate` (−0.60) and for `instrumental_artifact` (+0.50); `None`
   unless ≥20 out-of-transit cadences are available. See
   `docs/SCORING_MODEL.md §25`. The version 0.2.81 release gate passed
   2,883 default tests plus Ruff/mypy as 10/10 supervised gates in
   26.1 seconds under the canonical 6x6 topology.
   The 0.2.60 release passed 2,751 default tests plus Ruff/mypy as
   8/8 supervised gates under the 6×6 topology in 25.2 seconds.

## Milestone 1 — Scoring and Classification Engine ✓ COMPLETE

- [x] `schemas.py` — typed Pydantic data contracts
- [x] `features.py` — 35+ normalized feature extraction functions
- [x] `hypotheses.py` — Bayesian log-score models for 6 hypotheses
- [x] `priors.py` — versioned conservative default and mission-specific prior profiles
- [x] `scoring.py` — softmax posterior + configurable priors + FPP, detection confidence, novelty, habitability
- [x] `pathway.py` — submission pathway classifier (SCORING_MODEL.md §11)
- [x] CI via GitHub Actions (ruff → mypy → pytest)
- [x] `CLAUDE.md` — project context for AI coding agents

---

## Milestone 2 — Data Pipeline ✓ COMPLETE

- [x] `fetch.py` — query MAST via Lightkurve; return LightCurve + provenance metadata
- [x] `clean.py` — NaN removal, sigma-clip, normalization, detrending
- [x] `search.py` — BLS search → `CandidateSignal` list; iterative masking for multi-planet
- [x] `vet.py` — compute `RawDiagnostics` from light curve + signal; call `extract_features()`
- [x] `@pytest.mark.integration_live` tests against real MAST data

---

## Milestone 3 — End-to-End Validation ✓ COMPLETE

- [x] `notebooks/pipeline_demo.ipynb` — TOI-700 (TIC 150428135) full pipeline walkthrough
- [x] All 6 stages covered: Fetch → Clean → Search → Vet → Score → Classify
- [x] Human-readable candidate report rendered as Markdown in notebook
- [x] Figures: raw vs. cleaned flux, phase-folded transit, posterior bar chart, all-signals grid

---

## Milestone 4 — Calibration ✓ COMPLETE

- [x] `calibration.py` — reliability curves, Platt scaling (scipy), isotonic regression (PAVA)
- [x] One-vs-rest calibration per hypothesis; renormalized to sum to 1.0
- [x] Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
- [x] `Skills/train_xgboost.py` — includes post-training Platt calibration step

---

## Milestone 5 — Reporting ✓ COMPLETE

- [x] Rich-formatted candidate report via `exo <TIC-ID>` CLI
- [x] JSON output via `--output`
- [x] Scorer selection via `--scorer [bayesian|xgboost|ensemble|cnn|full-ensemble]`, `--model-path`, and `--cnn-checkpoint`

---

## Milestone 6 — Injection-Recovery Tooling ✓ COMPLETE

- [x] `Skills/injection_recovery.py` — inject synthetic box transits, recover via BLS
- [x] Measures recovery rate by radius, period, noise level
- [x] 25 tests in `tests/test_injection_recovery.py`
- [x] Produce and commit bounded production-pipeline recovery curves on real-background
  canonical cases, with manifest IDs and explicit linkage to the frozen
  `benchmark_cnn_v1` evidence package (`production_sensitivity_v1`)
- [x] Expand bounded sensitivity coverage to TTV, single-transit, data-gap,
   stellar-variability, multi-quarter, and longer-period cases before making a
  general completeness claim. Version 0.2.45 adds the explicit 16-trial v2
  scenario contract across two Q1-Q4 backgrounds, including event-overlap
  recovery semantics for single transits and cadence accounting for gaps. A
  cache-only merged-code run completed in 7.98 seconds with 8/16 recoveries and
  zero failures; durable evidence is `production_sensitivity_v2.json`. This
  checks scenario coverage, not general survey completeness.

---

## Milestone 7 — ML Ensemble Scorer ✓ COMPLETE

- [x] Tier 1 — XGBoost on tabular features (`ml/xgboost_scorer.py`, 45 tests)
- [x] Tier 3 — Stacking scorer blending XGBoost + CNN + Bayesian (`ml/stacking_scorer.py`, 22 tests)
- [x] Kepler training pipeline (`Skills/fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py`)
- [x] TESS training pipeline (`Skills/fetch_tess_toi.py`, `build_tess_training_data.py`)
- [x] Evaluation framework (`Skills/evaluate_scorer.py`, ROC-AUC, F1, reliability diagrams)
- [x] Combined training data (`Skills/build_combined_training_data.py`)
- [x] Offline CNN snippet split assembly (`Skills/build_cnn_training_data.py`, 13 tests)
- [x] Offline CNN split validation (`Skills/cnn_split_validator.py`, 15 tests)
- [x] Tier 2 scaffolding — CNN scorer wrapper, training loop, checkpoint/calibration helpers, phase-folded snippet wiring, and `cnn/full-ensemble` CLI modes
- [x] First Tier 2 candidate — trained and evaluated on the deterministic seed-42 split; rejected because held-out AUC was 0.7404 and calibration worsened Brier/ECE
- [x] T1-1 source-contract reset — verified public NASA/MAST sources, committed source snapshots, leakage-safe manifests, bounded raw-FITS cache policy, and storage estimates
- [x] Master Kepler corpus and checkpoint — trained `checkpoints/cnn_t1_1_kepler_master/best.pt` from the combined KOI+DR24 corpus; held-out gates passed with raw test AUC 0.9572, calibrated F1 0.8347, Brier 0.0580, ECE 0.0142, and T=1.0
- [x] Production Tier 2 promotion readiness — complete the new Astrometrics-policy evidence package before copying any checkpoint into `models/`
  - [x] Fix promotion tooling to accept the current temperature-scaling calibration JSON, not only legacy Platt fields, and print the required intentional checkpoint `git add -f`
  - [x] Add a model card for the master checkpoint
  - [x] Add a reproducibility manifest linking source snapshots, manifests, splits, config, calibration, metrics, SHA-256, runtime, and MPS/Python assumptions
  - [x] Add `data_selection/data_role_registry.yaml` for training, validation, calibration, and frozen-eval roles
  - [x] Mark the promoted architecture/data/preprocessing combination as the frozen `benchmark_cnn_v1` measuring stick
  - Preserve raw FITS as re-downloadable cache only; commit only selected production artifacts after explicit human approval
- [x] Human-approved CNN artifact promotion — approved on 2026-07-09; `benchmark_cnn_v1` is registered under `models/registry.json` and selected artifacts are promoted under `models/cnn/benchmark_cnn_v1/`
- [x] T1-2 stacking calibration — completed 2026-07-10 on 588 held-out K2 examples; calibrated weights (XGBoost=0.95/CNN=0.00/Bayesian=0.05) are wired into production

---

## Milestone 8 — Background Automation ✓ COMPLETE

- [x] `background/` module — SQLite-backed durable state (run ledger, reviewed/needs-follow-up logs, follow-up tests, reports, approvals)
- [x] `background/runner.py` — `background_run_once()` — one-shot scheduler-friendly invocation
- [x] `background/priority.py` — composite priority scoring (8 factors) with reason codes
- [x] `background/storage.py` — `BackgroundStore` with schema v2 tables for the run ledger, priority evaluations, outcomes, follow-up tests, reports, approvals, locks, and migrations
- [x] `background/reports.py` — draft Markdown/HTML reports; human-approval gate enforced
- [x] `background/fixtures.py` + `fixtures/known_tess_examples.json` — deterministic offline target pool
- [x] CLI subcommands: `exo background-run-once`, `run-summary`, `sqlite-integrity`, `target-priority-summary`, and 13 others
- [x] Scheduler docs (`docs/SCHEDULER.md`): cron, launchd, systemd timer examples
- [x] System profile (`docs/SYSTEM_PROFILE.md`): hardware sizing and batch-run defaults
- [x] `configs/background_search_v0.json` — versioned, fingerprinted configuration
- [x] 16 tests in `tests/test_background_automation.py`

---

## Milestone 8b — Star Scanner ✓ COMPLETE

- [x] `Skills/star_scanner.py` — `priority_score()`, `ScanLog` (JSON), `select_targets()` (TIC query), `scan_star()`, `run_background_scan()`
- [x] Priority scoring: Tmag (0.30), Teff/stellar type (0.25), sector coverage (0.25), contamination ratio (0.20)
- [x] TOI exclusion at startup; already-scanned exclusion via log; graceful Ctrl-C resume
- [x] 38 tests in `tests/test_star_scanner.py`

---

## Milestone 9a — Provenance Score ✓ COMPLETE

- [x] `compute_provenance_score(provenance: FetchProvenance) -> float` in `fetch.py`
- [x] Wired into `run_pipeline()` in `cli.py`; `provenance_score` included in JSON output rows
- [x] `tfop_ready` pathway now correctly enabled/blocked based on cadence, sector count, pipeline quality
- [x] 15 unit tests in `tests/test_fetch.py`; 4 flow tests in `tests/test_cli.py`
- [x] Documented in `docs/SCORING_MODEL.md §21`

---

## Milestone 9b — Candidate Ranking ✓ COMPLETE

- [x] `Skills/rank_candidates.py` — composite rank score weighting FPP, detection confidence, novelty, provenance, pathway
- [x] `load_candidates()`, `compute_rank_score()`, `rank_candidates()`, `print_rank_table()`
- [x] 12 tests in `tests/test_rank_candidates.py`

---

## Milestone 9c — Batch Scan ✓ COMPLETE

- [x] `Skills/batch_scan.py` — scan TIC ID lists from text/CSV; incremental JSON output with `--resume`
- [x] `read_tic_ids()`, `batch_scan()` with mock-injectable pipeline function
- [x] 14 tests in `tests/test_batch_scan.py`

---

## Milestone 9d — Sector Coverage ✓ COMPLETE

- [x] `Skills/sector_coverage.py` — query available TESS sectors per target without downloading data
- [x] `get_sector_coverage()`, `format_coverage_table()`; CLI with `--json` output
- [x] 10 tests in `tests/test_sector_coverage.py`

---

## Milestone 10a — Depth Scatter Chi-Square ✓ COMPLETE

- [x] `depth_scatter_chi2_score(depths, errors, chi2_threshold=3.0)` in `features.py`
- [x] New `depth_scatter_chi2_score: OptScore` field in `CandidateFeatures` (schemas.py)
- [x] Error-weighted reduced chi-square test complements existing robust-CV `depth_consistency_score`
- [x] Wired into `log_score_instrumental()` (+0.90 weight) and `log_score_planet()` (−0.60 weight)
- [x] 8 tests in `tests/test_features.py`; 5 tests in `tests/test_hypotheses.py`

---

## Milestone 10b — Phase-Fold Plots ✓ COMPLETE

- [x] `Skills/plot_lc.py` — `phase_fold()`, `plot_candidate()`, `plot_all()`
- [x] Generates PNG for each candidate row from `exo --output` JSON
- [x] No-op when matplotlib is absent; 11 tests in `tests/test_plot_lc.py`

---

## Milestone 10c — Watchlist + Summary Report ✓ COMPLETE

- [x] `Skills/watchlist.py` — atomic JSON watchlist; `add`, `remove`, `contains`, `list_ids`, `entries`, `clear`, `summary`; 13 tests
- [x] `Skills/summary_report.py` — `load_results`, `build_report`, `write_report`; partitions by status; candidates sorted by FPP; 14 tests

---

## Milestone 11a — Transit Timing Variation Score ✓ COMPLETE

- [x] `transit_timing_variation_score(midpoints, period_days, epoch_bjd, rms_threshold_minutes=10.0)` in `features.py`
- [x] New `transit_timing_variation_score: OptScore` field in `CandidateFeatures` (`schemas.py`)
- [x] O-C residuals in minutes; score = `clip(RMS_OC / threshold)` — saturates at threshold
- [x] Wired into `log_score_planet()` (−0.50 weight) and `log_score_instrumental()` (+0.60 weight)
- [x] 10 tests in `tests/test_features.py`; 3 tests in `tests/test_hypotheses.py`
- [x] Documented in `docs/SCORING_MODEL.md §22`

---

## Milestone 11b — TOI Checker + Export Candidates ✓ COMPLETE

- [x] `Skills/toi_checker.py` — `check_toi(tic_id)` queries ExoFOP TOI CSV; `format_toi_result()` one-liner status; handles column-name variations
- [x] `Skills/export_candidates.py` — `to_csv()`, `to_markdown_table()`, `to_summary_stats()`; 10-column export with display headers
- [x] 12 tests in `tests/test_toi_checker.py`; 13 tests in `tests/test_export_candidates.py`

---

## Milestone 11c — Alert Filter + Skills Guide ✓ COMPLETE

- [x] `Skills/alert_filter.py` — `filter_candidates()` AND-logic threshold filter (FPP, pathway, signals, rank, SNR); `apply_filters()` loads + writes JSON
- [x] `_fpp()` helper handles all dict shapes: `scores.false_positive_probability`, `best_fpp`, top-level
- [x] `docs/SKILLS_GUIDE.md` — workflow reference plus current inventory for 249 Skills; CLI examples; library usage pattern; ML pipeline walkthrough
- [x] 12 tests in `tests/test_alert_filter.py`

---

## Milestones 12-18 — Diagnostic And Operations Expansion ✓ COMPLETE

- [x] Milestone 12 diagnostic scores, CLI metadata, notebook generation, target prioritization, candidate comparison, timelines, and FITS helper utilities
- [x] Milestone 13 follow-up preparation, false-positive vetting, data quality, detrending comparison, recovery completeness, and HTML report utilities
- [x] Milestone 14 caching, period aliases, multi-planet checks, centroid analysis, catalog crossmatch, transit modeling, candidate database, follow-up scheduling, config management, alerts, and scorecards
- [x] Milestone 15 light-curve statistics, depth correction, nearby-star checks, binned exports, uncertainty, timing, candidate merging, multi-sector stacking, metadata, notes, TOI watching, contamination correction, benchmarking, and phase plots
- [x] Milestone 16 transit analysis and follow-up preparation utilities including radius, odd/even, secondary eclipse, momentum dump, duplicate TOI, activity, RV, impact parameter, observing request, ephemeris uncertainty, photometry combination, transit windows, labelled snippets, CNN augmentation, and report cards
- [x] Milestone 17 geometry, noise, period analysis, visibility, ground-truth matching, scatter metrics, centroid checks, and evidence aggregation utilities
- [x] Milestone 18 observability and analysis utilities including equilibrium temperature, TSM, airmass, moon separation, telescope time, false alarm probability, chi-square period checks, deduplication, run diffs, FITS exports, asymmetry scoring, trapezoid comparison, leaderboard, email formatting, and transmission-window prediction

---

## Future

- [x] Calibrate full-ensemble weights on a held-out set after CNN promotion — completed 2026-07-10 on 588 K2 examples; production weights are XGBoost=0.95, CNN=0.00, Bayesian=0.05.

---

## Milestone 19a — Multi-Sector Phase-Fold Comparison ✓ COMPLETE

- [x] `Skills/multi_sector_phase_compare.py` — offline per-sector phase-fold comparison for transit depth and phase centroid consistency
- [x] Flags insufficient coverage, weak/inverted signals, depth mismatches, and phase shifts conservatively
- [x] 12 tests in `tests/test_multi_sector_phase_compare.py`

---

## Milestone 19b — Static Candidate Dashboard Foundation ✓ COMPLETE

- [x] `docs/DASHBOARD_SPEC.md` — local-first dashboard data contract and guardrails
- [x] `Skills/candidate_dashboard_export.py` — static HTML dashboard from existing local candidate JSON rows
- [x] Preserves false-positive evidence, negative evidence, missing-score states, and blocking issues
- [x] Renders optional local phase-fold plot artifacts when supplied
- [x] 23 tests in `tests/test_candidate_dashboard_export.py`

---

## Milestone 19c — Local Read-Only Candidate API ✓ COMPLETE

- [x] `docs/API_SPEC.md` — local API contract, endpoints, guardrails, and non-goals
- [x] `Skills/candidate_api.py` — standard-library read-only HTTP API for local candidate JSON rows
- [x] Endpoints: `/health`, `/summary`, `/candidates`, `/candidates/<id>`, `/dashboard`, `/background/summary`, `/background/latest`
- [x] Static local review bundle endpoint: `/artifact.json`
- [x] Opt-in CORS headers for separate local frontends (`--cors-origin`)
- [x] Optional background SQLite summaries are read-only and do not create or mutate runtime databases
- [x] Candidate payloads carry optional phase-fold plot artifact paths
- [x] 33 tests in `tests/test_candidate_api.py`

---

## Milestone 19d — Interactive Local Candidate Browser ✓ COMPLETE

- [x] `Skills/candidate_browser_ui.py` — dependency-free browser UI for candidate review
- [x] Supports embedded-data mode for offline file viewing and API mode for `candidate_api.py`
- [x] Includes search, risk filtering, pathway filtering, summary metrics, detail panel, and optional plot previews
- [x] 20 tests in `tests/test_candidate_browser_ui.py`

---

## Milestone 20 — Slick Animated Command-Line UI ✓ COMPLETE

Unlocked by the formal production-ensemble acceptance PASS in version 0.2.33.
This milestone improves operator clarity without changing scientific scoring,
thresholds, or classifications. Version 0.2.96 implements it.

- [x] Add a polished terminal presentation for Fetch → Clean → Search → Vet → Score → Classify, with animated progress and useful elapsed-time status. `run_pipeline()` gained an optional `on_stage(stage_name)` callback fired before each of fetch/clean/search/vet_score_classify (vet/score/classify are combined into one displayed phase since they always run together per-signal); `exo scan`'s new `_StageAnimator` drives a `rich.console.Console.status()` spinner from it. **Scoping note**: this delivers elapsed-time only, not a predictive ETA — a single scan has no per-stage duration history to extrapolate from (unlike `batch_scan.py`'s N-item-rate ETA), so a fabricated estimate would violate the project's No-Unsupported-Completion-Claims policy rather than serve the operator.
- [x] Preserve stable machine-readable JSON output, exit codes, redirected output, and non-TTY/CI behavior — `--output` JSON, exit codes, and the post-run candidate summary are identical whether the spinner ran or not; regression-tested.
- [x] Provide an explicit `--no-animation`/reduced-motion path and graceful interruption/error rendering — `--no-animation` forces plain `[<elapsed>s] <Stage> ...` lines; animation also auto-disables whenever `sys.stdout.isatty()` is false. `KeyboardInterrupt` during a scan stops the spinner cleanly, prints `Interrupted during stage: <Stage>` to stderr, and exits `130` with no partial JSON written.
- [x] Add terminal-width, TTY/non-TTY, failure-path, and interruption regression tests — 15 new tests across `TestRunPipelineOnStage`, `TestScanCommand`, and `TestStageAnimator` in `tests/test_cli.py`, including a narrow (`width=20`) non-interactive-console render check. (A 16th test asserting on rendered `--help` text was added then removed after a CI-only failure showed Typer's Rich-based help renderer wraps long option names at widths this sandbox doesn't reproduce and doesn't honor `COLUMNS` the way Click's plain formatter does; the flag's registration is already proven end-to-end by the `--no-animation` invocation test.)
- [x] Document the interactive and automation-safe CLI modes — see `docs/DISCOVERY_RUNBOOK.md`'s "`exo scan` interactive vs. automation-safe display modes" section.

---

## Milestone 21 — Persistent EXO-Hunter Slash Terminal (COMPLETE)

- [x] Register `ExoHunter` as the persistent shell entry point.
- [x] Expose `/New-Search`, `/Follow-Up-Search`, `/Run-Search`,
  `/Show-Follow-Ups`, `/Help`, and `/Exit`, plus the existing import/recheck
  operations.
- [x] Delegate to the canonical one-shot functions instead of duplicating
  selection, scoring, execution, tables, or persistence.
- [x] Add persistent history, Tab completion, useful errors, and explicit
  `/` discovery.
- [x] Tie orbit/transit frames to real canonical work and disable motion for
  redirected I/O, CI, reduced-motion settings, `TERM=dumb`, and
  `--no-animation`.
- [x] Preserve scriptable, machine-readable operation through repeatable
  `--command`, `--script`, and unchanged canonical `--json` stdout.
- [x] Prove real new and follow-up create-through-execution workflows and clean
  schema-v6 integrity in `hunter_live_acceptance_v13.json`.
- [x] Pass PR CI, squash-merge, synchronize `main`, and rerun merged-main gates.

---

## Milestone 22 — Exact Shared Hunter Command Contract (COMPLETE)

- [x] Register `EXO-Hunter` while retaining `ExoHunter` as a compatibility alias.
- [x] Accept canonical `/Create-New-Search --targets N --mode ...` and
  `/Run-New-Search` commands without duplicating business logic.
- [x] Retain `/New-Search`, `/Follow-Up-Search`, and `/Run-Search` conveniences.
- [x] Add focused packaging, help-discovery, and exact-delegation tests.
- [x] Pass full gates, PR CI, squash merge, installed-command probes, and
  merged-main acceptance.

---

## Decision Tree (current implementation)

```
known_object posterior ≥ 0.80      → known_object_annotation
FPP ≥ 0.70                         → github_only_reproducibility
transit_count < 2                  → planet_hunters_discussion

TESS:
  all 9 tfop conditions met        → tfop_ready
  detection_confidence ≥ 0.45      → planet_hunters_discussion
  otherwise                        → github_only_reproducibility

Kepler/K2:
  p_planet ≥ 0.65, novelty ≥ 0.70,
  FPP ≤ 0.35                       → kepler_archive_candidate
  otherwise                        → github_only_reproducibility
```

See `src/exo_toolkit/pathway.py` and `docs/SCORING_MODEL.md §11` for full threshold values.
