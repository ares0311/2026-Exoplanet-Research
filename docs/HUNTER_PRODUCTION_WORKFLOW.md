# EXO-Hunter Production Workflow

## Production outcome

Version 0.3.3 closes the reviewed-prior-result bridge left open by version 0.3.2.
The shell entry points share one durable SQLite system of record and can run
without an AI model. Hunter acceptance is PROD on version 0.3.11. The recomputed
production snapshot is in `hunter_live_acceptance_v5.json`; the adaptive-
selection delta is in `hunter_live_acceptance_v8.json` (which supersedes
`hunter_live_acceptance_v7.json`); `hunter_live_acceptance_v9.json` adds live
create-through-follow-up-registration execution evidence on a real target
(additive to v8, not a correction). `hunter_live_acceptance_v10.json` closes
the one gap v9 explicitly left open: a real live 126-tile MAST sweep exercising
the new-mode adaptive-discovery-expansion loop itself, not just an offline
double. Execution (light-curve fetch) for the freshly-discovered targets did
not complete in that session — a sandbox filesystem write restriction on
`~/.lightkurve/cache` specific to that interactive agent session, not a code
defect (the failure was recorded correctly per Fail Loudly, not swallowed).
`hunter_live_acceptance_v11.json` then closed this fully: Lightkurve honors
`XDG_CACHE_HOME` if `$XDG_CACHE_HOME/lightkurve` already exists, so a
repo-local, git-ignored cache directory was pre-created and the same pending
search was re-run with that env var set (no source change). All 5 freshly-
discovered targets fetched real light curves, ran BLS/vet/Bayesian scoring,
and reached `candidate_found`, registering 16 new follow-up recommendations.
The full directive-required "New targets" scenario — discover broadly,
adaptively expand, create the exact durable search, execute exact targets,
persist real typed results/provenance, update follow-up eligibility — is now
demonstrated live end to end. See AGENTS.md's Python Environment Policy for
the reusable sandbox notes this session recorded.

**Version 0.3.10** corrects the follow-up selection contract itself, not just
its identity. Versions 0.3.3-0.3.9 all treated "zero targets clear the strict
FPP<0.15/confidence>0.40/pathway gate" as a valid terminal outcome and had
`create_search()` raise and create nothing whenever fewer candidates than
requested cleared that gate, in either `new` or `follow-up` mode. That is a
business-requirement violation, not a scientific finding: a normal top-N
request must return the best available N, with absolute quality reported
separately, and must fail only when fewer than N valid candidates exist at
all. `follow_up_universe()` now treats availability (not already scheduled,
completed, or revisit-deferred; a real unfollowed `candidate_found` signal
exists) as eligibility; the strict FPP/confidence/pathway bar is reported
per-candidate as `meets_strict_follow_up_bar` instead of removing candidates
from the selectable pool. Verified against a disposable copy of the real
production database: the 202-target durable follow-up universe now reports
190 available candidates (zero under the old gate), and requesting 10
follow-up targets returns exactly 10 best-available targets instead of
raising. See `artifacts/manifests/hunter_live_acceptance_v7.json`.
**Version 0.3.11** closes that remaining gap: `_select_live_new_candidates()`
no longer raises when the fixed-magnitude-window sweep returns fewer raw
candidates than requested. It widens the Tmag window (up to 3 retries, ±1.0
mag per step, clamped to [4.0, 18.0]) and retries before handing whatever it
found to `create_search()`, which alone decides whether to return fewer than
N or fail closed on genuine zero. Every attempt (Tmag range, raw candidate
count) is recorded in `selector_log.discovery_expansion_attempts`. The
126-tile grid itself remains a fixed, documented ~99 sq deg sample; this
widens the magnitude window per tile, not tile coverage. Originally verified
offline only against a scripted discovery double (see
`artifacts/manifests/hunter_live_acceptance_v8.json`); a later session then
ran it live against real MAST data (see `hunter_live_acceptance_v10.json`
above) — a deliberately narrow Tmag 4.0-4.1 window returned 0 raw candidates
from a real 126-tile sweep, the loop widened it to 4.0-5.1, and a second real
sweep found 13, enough to satisfy the requested 5 with zero shortfall.
**Version 0.3.12** fixes two real gaps a post-acceptance audit found in
`_select_live_new_candidates()` itself. First, its sufficiency check compared
the *raw* sweep count against the target count before known-variable
exclusion and QLP-product-availability filtering ever ran (those only inspect
the raw sweep's top `targets * 3` rows), so a sweep that cleared the raw
threshold but left the *eligible* pool short of N never triggered further
expansion. The check now runs after stage-two inspection, on the eligible
count, and `stage_two_goal` effectively grows because expansion itself
continues (widening Tmag range and tile coverage) whenever eligible
candidates fall short — not just when the raw sweep is thin. Second, the
126-tile grid was permanently fixed regardless of how many expansion retries
ran; `Skills/star_scanner.py` now has a second, disjoint 180-tile expansion
ring (`_EXPANSION_SEARCH_CENTERS` — interleaved offset bands plus two polar
bands, `TOTAL_SEARCH_TILES = 306`), and `_select_live_new_candidates()` widens
`max_tiles` (bounded, +60 tiles per retry, capped at the scanner's declared
total) alongside the existing Tmag widening, both logged per attempt in
`selector_log.discovery_expansion_attempts` (`max_tiles`,
`eligible_candidates`) and as `final_max_tiles`. Verified offline with
scanner doubles exercising both fixes (a sweep that returns enough raw rows
but too few eligible ones must still keep expanding; tile coverage must grow
and cap correctly) — no live 126+-tile MAST sweep was run from this session.

**Version 0.3.13** closes a real bypass of "no production command bypasses
the canonical optimizer or durable pipeline": `Skills/star_scanner.py`'s and
`Skills/batch_scan.py`'s standalone CLI scan modes wrote only to a local
`ScanLog`/JSON results file, invisible to Hunter's durable
`target_search_history`. `src/exo_toolkit/hunter_history.py` gains
`build_manual_scan_source()`, which builds one history-manifest "source"
from a completed manual scan's log file and result entries using exactly
the schema `HunterStore.import_history_manifest()` already verifies for the
seven curated legacy-log imports (byte-for-byte source hash, typed entries)
— no manifest file needs to be written to disk; the dict is passed straight
to `import_history_manifest()`. Both CLIs now call this after a real scan
(new `--hunter-db`/`--no-hunter-bridge` flags, bridging on by default):
`star_scanner.py`'s `--target` and default background-scan modes (via a new
`scanned_this_run` field in `run_background_scan()`'s return, additive to
the existing contract) and `batch_scan.py`'s CLI (via a new optional
`new_entries` out-parameter on `batch_scan()`, so `--resume` runs bridge
only the freshly-scanned targets, not the full accumulated results). Per
the plan, library-import usage by Hunter itself
(`_select_live_new_candidates()`'s calls into `select_targets()`/
`inspect_target_products()`) and the shard launcher's
`--execute-prepared-batch` path are untouched — the bridge is wired only
into the two real standalone-scan CLI entry points, verified by dedicated
tests proving `--execute-prepared-batch` and `--prepare-only` never call it.
A real integration bug was caught before merge: the manifest builder and
`import_history_manifest()` must resolve the same relative `source_path`
against the *same* root, or a log file living outside the repo tree (or a
CWD that differs from where the file was built) causes a fail-closed hash
mismatch; both bridges now explicitly pass the log file's own parent
directory as `source_root` to both calls rather than relying on either
function's independent default. Verified offline: real `ScanLog`/
`batch_scan()` entries round-trip through `import_history_manifest()` into
`HunterStore.searched_target_ids()`/`target_history()`.

**Version 0.3.14** adds the smart MAST recheck for deferred follow-ups named
in the plan as the last remaining gap. `follow_up_registry` gains two
nullable columns, `last_known_sectors` (JSON-encoded sorted sector list) and
`last_mast_checked_at` (ISO-8601), via the same ALTER-TABLE-if-missing
migration pattern already used for `revisit_reason`/`parent_follow_up_id`.
`HunterStore.record_sector_recheck()` is the only place that ever writes
them: it compares a fresh sector list against the row's last recorded
baseline (not a blind timer, and not the original deferral snapshot) and
flips the row from `deferred` back to `open` only when it finds sector
numbers not already known; a row with no baseline yet is treated as an
empty one, matching every follow-up this project has actually deferred so
far (deferred specifically because zero usable sectors existed). The
baseline/timestamp always advance to the check's result whether or not it
grew, and a `follow_up_events` row is appended either way (a `deferred`
audit entry when unchanged, an `open` transition when it flips) — no silent
no-op. The new `Recheck-Follow-Ups` shell entry point
(`exo_toolkit.hunter_cli:recheck_follow_ups_entry`) iterates every deferred
row with bounded concurrency (`--workers`, default 6), reusing
`Skills/sector_coverage.py`'s `get_sector_coverage()` (a metadata-only
`lightkurve.search_lightcurve` query, zero downloads) via the same
`_load_project_skill()` pattern the other Skill-reuse call sites already
use; a per-target query failure is recorded and reported (non-zero exit)
without blocking the rest of the batch. Verified offline against a real
no_data-deferred follow-up row built through the full create-execute
lifecycle (not a scripted double): a recheck finding new sectors flips it
open, one finding nothing leaves it deferred and still advances the
baseline, and a row already holding a non-empty prior baseline only flips
on sectors beyond that baseline, not the ones already known. No live MAST
query was run from this session.

**Version 0.3.15** adds an explicit `--history-source-root` override for
`Create-New-Search --history-manifest`, `HunterStore.validity_summary()`,
`validate_hunter_database()`, and `Skills/validate_hunter_acceptance.py`'s
`--history-source-root`. Root cause: `load_verified_history_manifest()`
(`hunter_history.py`) already accepted a `source_root` parameter, but every
caller above it left it unset, so a manifest's `source_path` entries always
resolved through `_repository_root_for()` -- a heuristic that walks up from
the manifest's location looking for the nearest ancestor containing
`pyproject.toml`+`src/` and uses that as the root. This is correct for the
default in-repo manifest (the walk-up finds this repo's own root) and for a
manifest under a plain OS temp directory (no such ancestor exists, so it
falls back to the manifest's own directory), but it silently resolves to the
**wrong** root for a manifest that happens to sit inside some *other*
checked-out repo's subtree -- exactly the "isolated or scripted operation"
case `--history-manifest` is documented to support. A prior audit session
first surfaced this as an apparent regression (a dedicated regression test,
`tests/test_hunter_cli.py::test_default_follow_up_imports_and_ranks_durable_history`,
failed deterministically on a clean 0.3.14 tree) but deeper investigation
found the failure was specific to that interactive sandbox: pytest's
`tmp_path` fixture normally resolves under a plain OS temp directory (no
`pyproject.toml` ancestor, heuristic falls back correctly), but in that
sandbox `tempfile.gettempdir()` could not use the real system temp directory
and fell back to the process's CWD -- which was this repo's own root -- so
pytest's tmp dirs landed *inside* this repo and the walk-up escaped test
isolation to find this repo's `pyproject.toml`+`src/` instead of the
intended tmp directory. That means the "tests passing" claims recorded for
versions up to 0.3.14 were very likely accurate in real CI; the underlying
design fragility they exposed was real, but it was latent, not a shipped
regression. **Lesson for future agents:** before treating a locally
reproduced test failure as proof of a shipped regression, check whether the
failure depends on filesystem ancestry relative to the *sandbox's* temp/cwd
behavior specifically -- rerun with an explicit, hermetic root rather than
trusting an implicit heuristic's default, and say so explicitly if the two
diverge; do not silently launder a sandbox artifact into a false regression
claim, and do not silently launder a real fragility into "not a bug" either.
The fix keeps the existing heuristic as the *default* (`--history-source-root`
omitted, `history_source_root=None` throughout) so the default manifest path
is unchanged, and adds the explicit override so isolated/scripted callers
never have to depend on the heuristic at all. `test_default_follow_up_imports_and_ranks_durable_history`
now passes `--history-source-root` explicitly (hermetic, no longer
environment-dependent), and a new regression test,
`test_history_source_root_overrides_repo_root_walk_up_heuristic`, builds a
decoy nested repo (its own `pyproject.toml`+`src/`) around a manifest and
proves the walk-up resolves to the wrong root and fails closed without the
override, then succeeds with it.

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
schema-v1 provenance-complete manifest for isolated or scripted operation --
**pass `--history-source-root` alongside it** (see version 0.3.15 below); do
not rely on the repo-root walk-up heuristic to guess the right directory.
The candidate-pool count includes every evaluated prior target, including
ineligible rows, so any shortfall (selected < requested) is an auditable,
reported fact rather than an inferred one. `Create-New-Search` returns the
best available N and only fails outright when zero candidates are available
at all (version 0.3.10).

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

Recheck deferred follow-ups for new MAST sector coverage (metadata-only, no
downloads):

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Recheck-Follow-Ups --pipeline QLP --workers 6
```

`Recheck-Follow-Ups` never flips a deferred row back to revisit-eligible on a
blind timer — only when `Skills/sector_coverage.py`'s `get_sector_coverage()`
finds sector numbers not already recorded on that row's baseline. A row with
no baseline yet is treated as empty, so the first recheck of a row deferred
for zero usable sectors flips it as soon as any real sector appears. Every
recheck advances the row's `last_known_sectors`/`last_mast_checked_at`,
whether or not it flips, and appends a `follow_up_events` audit row either
way.

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

Version 0.3.9 gives each selection path a distinct immutable identity:
`exo_hunter_tic_v2`, `exo_hunter_follow_up_v2`, or
`exo_hunter_operator_candidates_v1`. Every manifest also freezes the complete
JSON ranking contract: formula, weights, information-gain definition,
eligibility thresholds, and accepted pathways. Candidate snapshots created by
the built-in selectors carry the same selector ID. The validity verifier fails
if candidate, manifest, or contract identities disagree, so a future ranking
change must receive a new selector version rather than silently reusing one.

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

Every detected candidate signal enters the follow-up registry (version 0.3.10):
priority is a continuous function of FPP and detection confidence, and each
row also records `meets_strict_production_bar` — whether it clears the
conservative bar (FPP below 0.15, detection confidence above 0.40, an eligible
pathway) — as informational evidence, not a filter on whether the row exists.
The recommendation never claims confirmation and asks for
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
