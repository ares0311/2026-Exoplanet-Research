# Local Candidate API Specification

## Purpose

Define the local read-only API for browsing candidate signals and follow-up
targets from existing project JSON artifacts.

The API is a review and inspection surface. It must not claim discovery,
confirmation, external validation, or submission approval.

## Initial Implementation

The first implementation uses only the Python standard library:

```bash
python Skills/candidate_api.py results.json --host 127.0.0.1 --port 8765
```

When a local background automation database already exists, it can be exposed
read-only:

```bash
python Skills/candidate_api.py results.json --background-db-path logs/background_search.sqlite3
```

This keeps the API dependency-free, local-first, and compatible with macOS and
portable Python environments.

## Data Source

The API reads the same local JSON shapes accepted by
`Skills/candidate_dashboard_export.py`:

- `exo --output` candidate rows
- `batch_scan.py` result rows
- ranked candidate rows from `rank_candidates.py`
- wrapped JSON with top-level `candidates` or `rows`
- candidate-database style rows after JSON export or conversion
- optional local phase-fold plot artifact paths such as `plot_path`,
  `phase_plot_path`, or `phase_fold_plot_path`

It does not query MAST, ExoFOP, Gaia, NASA Exoplanet Archive, or any other live
service.

## Endpoints

| Endpoint | Method | Output | Purpose |
| --- | --- | --- | --- |
| `/health` | GET | JSON | Local service health and candidate count. |
| `/summary` | GET | JSON | Aggregate candidate counts, FPP summary, risk counts, pathway counts. |
| `/candidates` | GET | JSON | Normalized candidate list. |
| `/candidates/<candidate_id>` | GET | JSON | One normalized candidate, or 404 if missing. |
| `/dashboard` | GET | HTML | Static dashboard view using the shared dashboard contract. |
| `/background/summary` | GET | JSON | Read-only aggregate summary for an optional background SQLite log. |
| `/background/latest` | GET | JSON | Latest background run alert state, reason, report paths, and approval state. |
| `/artifact.json` | GET | JSON | Single-file local review bundle with summary, candidates, and background status. |
| `/` | GET | JSON | Endpoint index. |

All mutation methods are rejected. The API is read-only.

## Response Guardrails

- Candidate rows are normalized through the same contract as the static
  dashboard.
- Missing scores remain `null` in JSON, not `0`.
- Blocking issues produce a `blocked` risk band.
- Optional phase-fold plot paths are carried through as review artifacts when
  they point to common local image formats.
- Responses include language guardrails describing rows as candidate signals or
  follow-up targets.
- No response should use a confirmed-discovery label for internally detected
  signals.
- External submission and live-service flags remain false.
- Background SQLite endpoints never create, initialize, migrate, or mutate a
  database. Missing databases are reported as unavailable.
- Background approval records are summarized for review context only; the API
  does not provide any external submission action.

## Non-Goals

- No authentication or multi-user state.
- No external submission endpoint.
- No live catalog queries.
- No writes back to source JSON.
- No replacement for `logs/background_search.sqlite3`.
- No SQLite writes, migrations, or scheduler actions.

## Future Extensions

- Browser UI consuming the JSON endpoints. Implemented by
  `Skills/candidate_browser_ui.py`.
- Static artifact mode for GitHub Actions or local report bundles. Implemented
  as `/artifact.json`.
- CORS configuration if a separate local frontend is introduced.
