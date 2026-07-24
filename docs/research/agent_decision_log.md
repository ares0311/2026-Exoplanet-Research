# Astrometrics Agent Decision Log — 2026 Exoplanet Research

Per `docs/astrometrics_coding_agents_master_guide.md`'s prescribed format.
Log decisions made in this repo that touch shared cross-project questions
raised by the master guide, so a sibling repo's agent can see what this repo
already resolved instead of re-deciding it.

## 2026-07-24 Decision

Repo: 2026 Exoplanet Research

Decision: Durable search/candidate history for this repo lives inside this
repo — local SQLite system of record
(`data/hunter_searches.sqlite3`, git-ignored) plus a portable, hash-verified
JSON export (`data_selection/hunter_prior_search_history_v1.json`,
`schema_version: 1`). No shared external database or shared
`astrometrics_ml_hardening` package was adopted. Full interface spec for
sibling repos: `docs/HUNTER_CROSS_PROJECT_INTERFACE.md`.

Why: The active Hunter production directive is explicit that "the three
Hunter repos are isolated. Do not assume cross-repo filesystem access" and
to "implement the smallest stable interoperable solution rather than
coupling repo internals" only if cross-project knowledge is actually
required for correct selection. Object-identity overlap only plausibly
exists with Technosignatures (shared TIC/KIC/EPIC stellar catalogs), not
Near Earth Objects (disjoint minor-planet identity space), so a full shared
package/database was judged more coupling than currently justified — a
documented, copyable file format plus a ~90-line verifier
(`src/exo_toolkit/hunter_history.py`) is the smaller solution and already
exists in production use for this repo's own legacy-log reconciliation.

Alternatives considered:
- Shared `astrometrics_ml_hardening` package (master guide's proposed shared
  implementation package) — rejected for now as more coupling than the
  current business need (avoiding duplicate target searches) requires.
- External/shared database — rejected for the same reason, plus it would
  require infrastructure none of the three repos currently run.
- Live cross-repo filesystem reads — attempted (this repo's
  `.claude/settings.local.json` sandbox `denyRead` was cleared for
  `2026 Technosignatures` and `2026 Near Earth Objects` on 2026-07-24) but
  found insufficient: this session's own tools independently enforce a
  "stay inside current git root" restriction not controlled by that config,
  confirmed by direct test (`Read`/`Bash` on a Technosignatures path both
  refused with "outside current git root" after the grant). Cross-repo
  exchange is human-mediated (file copy) until that separate restriction is
  addressed.

Citation: `docs/astrometrics_coding_agents_master_guide.md` §"Missing Items
/ Suggestions For The User", item 2; HUNTER PROD CLOSURE DIRECTIVE
(2026-07-24 session), "IMPLEMENTATION CONSTRAINTS" §cross-project knowledge.

Validation plan: revisit only if a real, observed duplicate-search cost
across projects is found (e.g. Technosignatures independently re-searching
a TIC target Exoplanet Research already has a durable result for). Until
then, the copyable-file contract in `docs/HUNTER_CROSS_PROJECT_INTERFACE.md`
is sufficient and no further coupling is planned.
