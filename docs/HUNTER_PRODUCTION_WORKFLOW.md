# EXO-Hunter Production Workflow

## Production outcome

Version 0.3.3 closes the reviewed-prior-result bridge left open by version 0.3.2.
The shell entry points share one durable SQLite system of record and can run
without an AI model. Hunter acceptance is PROD on version 0.3.8, with the
recomputed evidence in `artifacts/manifests/hunter_live_acceptance_v5.json`.
An earlier audit found that consumed actionable follow-ups remained open and
could be scheduled repeatedly. Version 0.3.5 adds durable scheduling/disposition events
and parent-child recommendation relationships. Version 0.3.6 makes
non-executable, revisit-gated recommendations canonically `deferred` rather
than `open`. The prior acceptance remains
preserved and is corrected by
`artifacts/manifests/hunter_live_acceptance_v2_reassessment.json`. Merged-main
schema-v4 migration and current archive evidence are recorded in
`artifacts/manifests/hunter_live_acceptance_v3.json`; replacement acceptance is
`artifacts/manifests/hunter_live_acceptance_v4.json`:

```text
candidate universe
→ identity/history resolution
→ eligibility
→ deterministic ranking
→ immutable manifest
→ pending search
→ resumable run attempts
→ acquisition/preprocessing/scoring
→ composite interpretation
→ append-only target history
→ evidence-based follow-up registry
→ recommended action
```

The Bayesian scorer remains the default. XGBoost, CNN, and ensemble modes are
optional and do not control target eligibility or ranking.

## Commands

Create a new-target search:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Create-New-Search --targets 100 --mode new
```

Create a follow-up search from the complete durable prior-search universe:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Create-New-Search --targets 10 --mode follow-up
```

The default source is
`data_selection/hunter_prior_search_history_v1.json`; it is imported
automatically and idempotently. `--history-manifest` may point to another
schema-v1 provenance-complete manifest for isolated or scripted operation.
The candidate-pool count includes every evaluated prior target, including
ineligible rows, so zero selected targets is supported by an auditable
universe rather than inferred from an empty registry.

Execute or resume the exact pending manifest:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/Run-New-Search --workers 6 --scorer bayesian
```

Show actionable follow-ups:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Show-Follow-Ups
```

Import a checksum-verified reviewed prior result before showing it:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Import-Follow-Up \
  --evidence-file artifacts/manifests/hunter_reviewed_followup_import_v1.json
```

`Import-Follow-Up` records the source as a completed historical manifest, run,
target-history row, and follow-up row. This evidence row explicitly carries
`search_eligible=false`; it remains visible in `Show-Follow-Ups`, but cannot
seed `Create-New-Search --mode follow-up` until their stated revisit condition
is satisfied. Source-file hash drift fails before any database mutation.

All commands support `--json` for scripts and `--no-color` for predictable
plain output. A candidate file may be supplied to `Create-New-Search` as JSON
or CSV. Follow-up rows must carry who searched, when, source project, method or
data, result, and a provenance URI; missing provenance fails before a search is
created.

## Durable entities

The default system of record is `data/hunter_searches.sqlite3`. It is local
runtime state and is ignored by git, like the existing live candidate ledgers.
Run Reports under `artifacts/manifests/run_reports/hunter_search.jsonl` make
completed attempt summaries GitHub-visible.

| Concept | Table | Mutation rule |
|---|---|---|
| Candidate catalog | `candidate_catalog` | Append one immutable snapshot per candidate per search creation |
| Search manifest | `search_manifests` + `search_manifest_targets` | Immutable target membership, ordering, config, selector version, and SHA-256 |
| Search run | `search_runs` | One durable row per attempt; interrupted attempts are marked explicitly on resume |
| Target search history | `target_search_history` | Append-only result, failure, and provenance events |
| Follow-up registry | `follow_up_registry` | Stable evidence-based entries with current disposition and explicit parent relationship |
| Follow-up lifecycle | `follow_up_events` | Append-only transitions linked to consuming searches |

`search_state_events` is append-only and records pending, running, interrupted,
partial, failed, and completed transitions. CSV is operator review output, not
the system of record. Searches of 100 targets or fewer render a terminal table;
larger manifests are exported under `reports/search_manifests/`.

## Validity and provenance acceptance gates

Hunter production acceptance is evidence-derived, not a declaration stored in
JSON. Every release that changes selection, execution, persistence, scoring, or
follow-up behavior must pass all of these gates on the exact tree being
accepted:

1. **Source lineage:** every imported source path exists and its bytes match
   the declared SHA-256 before any database mutation. A digest with valid syntax
   but unverified content is not accepted provenance.
2. **Typed execution provenance:** every target outcome, including unexpected
   failures and no-data/null results, preserves the candidate snapshot, pipeline
   configuration, code identity, scorer identity, and any acquisition context
   collected before the failure. Empty or runner-name-only provenance is invalid.
3. **Model and calibration identity:** model-backed runs preserve canonical
   paths plus SHA-256 digests for every XGBoost model, CNN checkpoint,
   calibration artifact, and score-context artifact actually used. The default
   Bayesian path records explicitly that no external model artifact was used.
4. **Relational validity:** the built-in integrity command runs SQLite integrity
   and foreign-key checks, required-table/schema checks, manifest and candidate
   snapshot hash recomputation, count reconciliation, exact-membership checks,
   run/history counter reconciliation, JSON decoding and typed row validation,
   and follow-up lifecycle reconciliation.
5. **Storage-enforced history:** manifest membership, candidate snapshots,
   target history, search-state events, and follow-up events are append-only at
   the SQLite layer. Normal application transitions may update only explicitly
   mutable attempt and follow-up disposition fields.
6. **Evidence-derived acceptance:** committed acceptance tests recompute hashes,
   counts, provenance completeness, lifecycle consistency, and source lineage
   from the committed contracts and a production database snapshot/copy. Tests
   must not treat a `"pass:"` string as proof of the underlying requirement.
7. **Deterministic selection validity:** ranking inputs remain finite, typed,
   and reproducible. New-target and follow-up selection record a named expected-
   information-gain/suitability component in addition to novelty, availability,
   scientific priority, and storage/compute cost.
8. **Independent reproducibility:** the three installed Hunter entry points run
   without AI or another repository, exact pending searches survive process
   restart, and partial/failed work remains loud and resumable.

The canonical verifier is `HunterStore.validity_summary()`. CLI creation and
execution fail closed when it reports any issue. Acceptance artifacts may
summarize a verified run, but the verifier output and exact-tree quality-gate
summary are the evidence. If any gate above is not implemented and passing, the
Hunter lifecycle is **PARTIAL**, not PROD.

Run the standalone verifier against a live database or a committed compressed
acceptance snapshot:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python Skills/validate_hunter_acceptance.py \
  --db data/hunter_searches.sqlite3
```

## Session closure loop

Before editing, reconstruct the current state from repository evidence rather
than a previous agent's plan. Maintain this checklist in the PR handoff:

- [ ] Last three PRs, their reviews, and recent relevant commits read
- [ ] Current behavior and durable database state verified
- [ ] Complete candidate-to-follow-up pipeline mapped
- [ ] Highest-priority validity or provenance gap identified
- [ ] Exact end-to-end path remains intact and restart-safe
- [ ] No AI dependency introduced into core logic
- [ ] Durable history and provenance preserved byte-for-byte
- [ ] Relevant focused and canonical tests pass on the exact tree
- [ ] Every PROD gate above checked from evidence
- [ ] PR handoff records decisions, tests, risks, and exact next work

After each meaningful work unit, re-run the verifier/tests, identify the next
highest-priority failing gate, and continue. A failed strategy must yield new
diagnosis and a changed approach; it must not be retried unchanged. Stop only
at verified acceptance or a genuine external blocker.

## New-target selection

The default metadata-only selector requests at least 10,000 TIC candidates and
performs two deterministic stages:

1. A full 126-tile TIC sweep ranks the broad pool by magnitude, stellar type,
   sector coverage, contamination, and stellar radius. TOIs, CTOIs, confirmed
   transiting hosts, and successfully searched EXO-Hunter targets are excluded.
2. The leading three candidates per requested slot receive strict ASAS-SN
   known-variable screening and QLP product-metadata inspection. Final ranking
   adds verified data availability and storage cost. Failed metadata checks,
   known variables, no-product targets, and candidates not advanced from the
   first stage remain frozen in the candidate catalog with explicit ineligible
   reasons; they can never enter the manifest silently.

Ranking is stable by score, canonical ID, and target ID. The manifest freezes
the archive-derived result so `Run-New-Search` never regenerates or substitutes
targets. `selector_log` records requested/returned pool size, tile failures,
coverage, stage-two count, exclusions, and eligible count.

## Execution and recovery

`Run-New-Search` uses six I/O workers by default on the recorded M4 Max. Every
worker runs the existing production pipeline, and only the parent writes
SQLite. Results preserve exact candidate snapshots, raw-product provenance,
pipeline context, code/scorer versions, individual signal scores, the strongest
composite result, interpretation, and failures.

The command returns nonzero for partial or failed execution. A later invocation
uses the same manifest, records an interrupted prior attempt if necessary,
skips only terminal target outcomes, and retries failed/unwritten targets. A
completed manifest cannot be executed again.

Candidate signals enter the follow-up registry only when the existing
conservative gate is met: FPP below 0.15, detection confidence above 0.40, and
an eligible pathway. The recommendation never claims confirmation and asks for
centroid/contamination, odd-even, secondary-eclipse, phase-fold, and additional
event-covering evidence where needed.

Selecting an actionable registry row atomically transitions it from `open` to
`scheduled` and records the consuming search ID. Failed attempts leave it
scheduled so only the exact search can resume. A terminal candidate/no-signal
outcome closes it as `completed`; a terminal no-data outcome marks it
`deferred` and non-executable. Any new recommendation records its source row as
`parent_follow_up_id`. Stale candidates fail the transaction instead of
creating a duplicate pending search. `Show-Follow-Ups --status all` exposes the
current state and complete event history in JSON.

Recommendations created with `search_eligible=false` begin in `deferred`, not
`open`; schema v4 migrates earlier rows append-only and preserves their revisit
reason. Consequently `open` means actionable throughout the CLI and durable
store instead of mixing actionable and waiting-for-data work.

## Known limits

- The 126 cone-search tiles cover about 99 square degrees, not the whole sky.
  The stored search log is authoritative about achieved coverage.
- Day-1 selection searches already-extracted light curves. Raw TESS FFI
  photometric extraction remains a separate future phase requiring explicit
  authorization; see `docs/DISCOVERY_RUNBOOK.md`.
- Public catalog and MAST metadata can change. Reproducibility applies to the
  frozen candidate/manifest snapshot, not to an assertion that a later live
  query must return identical catalog contents.
- No external submission or discovery claim is implemented or authorized.

## Verification

`tests/test_search_lifecycle.py` and `tests/test_hunter_cli.py` cover the full
offline lifecycle, including 100-from-10,000 selection, schema integrity,
manifest immutability, loud partial failure, restart/resume, history
preservation, external follow-up provenance, CSV review export, Run Report
injection, the three exact package entry points, and repository-Skill loading
from an installed process whose working directory is outside the repo.
Real-service acceptance used a live 10,000-candidate TIC universe and nine QLP
products for TIC 237884073. The original 0.3.1 result exposed a nested scorer-
schema defect; its history was preserved, and 0.3.2 corrected the consumer.
Neither acceptance run passed the unchanged FPP < 0.15 gate, so zero new rows
was scientifically correct for those runs but insufficient as live follow-up
creation evidence. Version 0.3.3 adds checksum-verified import of a separate,
already reviewed real result plus tests for migration, idempotency, provenance,
visible deferral, and selection exclusion. The merged-main import created one
real registry/history row; a repeated import returned the same stable IDs with
no duplicate rows. That evidence remains valid, but it did not exercise
consumption of an actionable row. Version 0.3.5 closes the resulting
scheduling/disposition defect. Version 0.3.6 closes the status-semantics gap
exposed by the first real v3 migration. The merged-main v4 production migration
is integrity-clean and the real row is now durably `deferred`. A 2026-07-21
metadata-only MAST check still found only sectors 1 and 28, both already proven
not to cover a predicted event. That leaves the real recommendation correctly
deferred; it is a normal scientific data state, not a production blocker.
Version 0.3.7 normalizes all seven preserved discovery logs into a committed,
checksum-provenance-stamped history manifest containing 608 events across 200
targets. Default follow-up creation imports those searches, runs, exact target
memberships, results, failures, and provenance idempotently before ranking the
complete durable history plus registry dispositions. Historical partial runs
remain visibly `archived_partial` and never masquerade as resumable work.
Merged-main production import preserved 608 historical events, produced a
202-target combined universe, and returned zero eligible targets with explicit
dispositions and no search creation. A repeat invocation changed no durable
row counts. Version 0.3.7 acceptance is therefore PASS. Version 0.3.8 replaces
digest-syntax and self-attested checks with source-byte verification, typed
execution and model-artifact identities, schema-v5 append-only triggers,
comprehensive content/relationship verification, and an executable acceptance
validator.
