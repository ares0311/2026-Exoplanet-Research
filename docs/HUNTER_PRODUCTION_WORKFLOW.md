# EXO-Hunter Production Workflow

## Production outcome

Version 0.3.2 closes the undocumented operator bridge between target selection
and the existing fetch → clean → search → vet → score → classify pipeline. The
three shell entry points now share one durable SQLite system of record and can
run without an AI model. Merged-main live acceptance is PASS; immutable evidence
is committed at `artifacts/manifests/hunter_live_acceptance_v1.json`:

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
| Follow-up registry | `follow_up_registry` | Stable evidence-based entries; duplicates are not rewritten |

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
schema defect; its history was preserved, 0.3.2 corrected the consumer, and a
provenance-linked follow-up run persisted the true minimum-FPP composite. No
signal passed the unchanged FPP < 0.15 gate, so zero live registry rows is the
correct scientific outcome rather than missing work. The qualifying
registration/recommendation branch remains covered by end-to-end store/CLI
tests using the same production nested score shape.
The version 0.3.2 release gate passes all 3,051 default tests plus Ruff, mypy,
the incomplete-implementation scan, and directive-integrity scan as 10/10
supervised gates under the canonical 6×6 topology.
