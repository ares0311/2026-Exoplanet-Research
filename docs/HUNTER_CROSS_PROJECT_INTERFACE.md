# Cross-Project Interface: What This Repo Publishes For Sibling Astrometrics Projects

Date: 2026-07-24

Audience: an agent or operator working in `2026 Technosignatures` or
`2026 Near Earth Objects` that has been granted (or is considering granting)
read access into this repo, or vice versa. If you landed here from one of
those repos, start reading now — this file exists specifically for you.

Context: `docs/astrometrics_coding_agents_master_guide.md` (shared across all
three repos) leaves "where durable data manifests and candidate ledgers
live: inside each repo, in a shared data repo, or in an external database"
as an open decision (its "Missing Items" §2). This file is Exoplanet
Research's answer for its own object-identity domain, recorded per that
guide's own decision-log convention in
[`docs/research/agent_decision_log.md`](research/agent_decision_log.md).

## Object identity this repo owns

Canonical target identity here is a stellar catalog ID: `TIC <n>` (TESS,
primary), or `KIC <n>` / `EPIC <n>` (Kepler/K2, legacy). See
`src/exo_toolkit/search_lifecycle.py` (`target_id`, `canonical_id` columns)
and `docs/HUNTER_PRODUCTION_WORKFLOW.md`.

- **Relevant to Technosignatures**: if that repo also observes stars (radio
  or optical technosignature search reuses the same TIC/KIC/EPIC catalogs),
  there is real object-identity overlap worth checking before either project
  treats a star as "never searched by any Astrometrics project."
- **Not relevant to Near Earth Objects**: NEOs use minor-planet/asteroid
  designations, a disjoint identity space from stellar catalog IDs. No
  object-identity bridge is expected or needed between this repo and that
  one; noted explicitly so no one spends effort building it.

## What this repo publishes, and where

- **System of record**: `data/hunter_searches.sqlite3` — local, git-ignored,
  append-only (see `src/exo_toolkit/search_lifecycle.py`). Not portable and
  not meant to be read directly by another repo.
- **Portable, hash-verified export**:
  `data_selection/hunter_prior_search_history_v1.json` — `schema_version: 1`,
  a list of `sources[]` (each with `source_path`, `source_sha256`,
  `provenance_uri`, `searched_by`, `mode`) plus per-target `entries[]`
  (`target_id`, `canonical_id`, `mission`, `status`, `searched_at`,
  `best_fpp`, `best_pathway`, ...). This is the file to read (or copy) if you
  want to know which TIC/KIC/EPIC targets Exoplanet Research has already
  searched.
- **Verifier/loader**: `src/exo_toolkit/hunter_history.py` —
  `load_verified_history_manifest()` is ~90 lines, fail-closed (rejects
  missing files, hash mismatches, malformed entries). It has no dependency on
  the rest of this package's internals. If a sibling repo wants to produce or
  consume the same shape, copy this file directly rather than re-deriving the
  design.

## Current operational limitation (read this before assuming access works)

On 2026-07-24, read access to `2026 Technosignatures` and
`2026 Near Earth Objects` was granted from this repo's
`.claude/settings.local.json` (cleared `sandbox.filesystem.denyRead` and the
matching `permissions.deny` `Read(...)`/`Bash(...)` rules for both paths;
`Edit`/`denyWrite` were deliberately left blocking).

That grant is **not sufficient by itself**. This session's own tools (Read,
Bash) independently enforce a "stay inside the current git root" restriction
that is separate from that sandbox config and was not lifted by editing it —
confirmed empirically the same day: both `Read` and `Bash` refused paths
under `2026 Technosignatures` with "outside current git root" even after the
settings edit. So an agent working in this repo still cannot actually read
sibling-repo files yet, despite the permission grant looking complete on
paper.

Until that harness-level restriction is separately addressed, treat
cross-repo exchange as **human-mediated**, not automatic:

1. If you're an agent in Technosignatures or NEO and want Exoplanet
   Research's search history: ask the operator to copy
   `data_selection/hunter_prior_search_history_v1.json` into your repo, then
   verify and load it with your own copy of `load_verified_history_manifest()`.
2. If you want Exoplanet Research to know about targets your repo already
   covered: publish an equivalently-shaped `schema_version: 1` file in your
   own repo and ask the operator to copy it into this repo's
   `data_selection/` directory — the existing reconciliation pattern (see
   `docs/DISCOVERY_RUNBOOK.md`'s superseded-notice section and
   `data_selection/hunter_prior_search_history_v1.json`'s `sources[]` list,
   which already reconciles seven legacy discovery logs the same way) applies
   unchanged to an external source.

## Non-goals

This is not a proposal for a shared database, a shared `astrometrics_ml_hardening`
package, or automatic live cross-repo sync. The Hunter directive is explicit
that the three repos are isolated and that a smaller interoperable solution
is preferred over coupling repo internals. Revisit only if real, observed
duplicate-search cost across projects justifies the extra coupling — record
that justification in `docs/research/agent_decision_log.md` if it happens.
