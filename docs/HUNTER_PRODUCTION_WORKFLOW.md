# EXO-Hunter Production Workflow

## Production outcome

Version 0.3.3 closes the reviewed-prior-result bridge left open by version 0.3.2.
The shell entry points share one durable SQLite system of record and can run
without an AI model. Hunter acceptance is temporarily PARTIAL after a fresh
audit found that consumed actionable follow-ups remained open and could be
scheduled repeatedly. Version 0.3.5 adds durable scheduling/disposition events
and parent-child recommendation relationships. The prior acceptance remains
preserved and is corrected by
`artifacts/manifests/hunter_live_acceptance_v2_reassessment.json`:

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

Create a follow-up search from the open follow-up registry:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Create-New-Search --targets 10 --mode follow-up
```

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
scheduling/disposition defect; merged-main migration and replacement
acceptance remain required.
