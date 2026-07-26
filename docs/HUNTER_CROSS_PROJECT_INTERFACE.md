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

## Current read-only mechanism

The 2026-07-25 verification established that sibling reads work when the path
is computed inside the running process relative to this checkout. The
write-side repository boundary remains enforced and is intentionally
unchanged.

`src/exo_toolkit/hunter_cross_project.py` provides the stable consumer:

- `sibling_history_export_path("technosignatures")` resolves
  `../2026 Technosignatures/data_selection/hunter_prior_search_history_v1.json`
  without a hard-coded absolute path.
- `Create-New-Search --cross-project-sibling technosignatures` prefers that
  normalized export. When it is absent, a strict read-only adapter normalizes
  the sibling's real `results/scan_history.ndjson` `prod_scan_history_v1`
  records in memory and independently verifies the exact source hash.
- `Create-New-Search --cross-project-history-path <file>` retains the portable
  operator-copied fallback. When origin source bytes are unavailable, the
  manifest is explicitly `stale-but-usable`, never silently `valid`.
- With neither flag, new-mode production imports the committed
  `data_selection/cross_project_imports/techno_hunter_history_v1.json`. If the
  sibling checkout is present, its real `results/scan_history.ndjson` is
  source-hash verified; otherwise the copy remains `stale-but-usable`.

Imported identities are append-only schema-v6 rows. TIC/HIP/KIC aliases affect
new-target eligibility, but Techno-Hunter's radio outcome vocabulary is
preserved verbatim and is never translated into an EXO-Hunter scientific
result. NEO-Hunter is intentionally absent because minor-planet designations
are a disjoint identity space.

At the time of verification, the real Technosignatures source history existed
and matched SHA-256
`c70b35120722754849c589db1d563060b3f4d6a32a246c25d16f2a116266b3fb`;
its advertised normalized export file did not yet exist. The direct sibling
flag was therefore verified through the strict raw-history adapter; the
default source-verified copied export remains the portable no-sibling
fallback.

## Non-goals

This is not a shared database, a shared package, cross-repo write path, or
automatic mutation of a sibling checkout. The interoperability surface is the
versioned JSON export, plus the narrowly validated Techno-Hunter
`prod_scan_history_v1` read adapter until the sibling publishes that export.
