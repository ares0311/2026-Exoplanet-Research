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
| `/` | GET | JSON | Endpoint index. |

All mutation methods are rejected. The API is read-only.

## Response Guardrails

- Candidate rows are normalized through the same contract as the static
  dashboard.
- Missing scores remain `null` in JSON, not `0`.
- Blocking issues produce a `blocked` risk band.
- Responses include language guardrails describing rows as candidate signals or
  follow-up targets.
- No response should use a confirmed-discovery label for internally detected
  signals.
- External submission and live-service flags remain false.

## Non-Goals

- No authentication or multi-user state.
- No external submission endpoint.
- No live catalog queries.
- No writes back to source JSON.
- No replacement for `logs/background_search.sqlite3`.

## Future Extensions

- Read-only background SQLite summary endpoints.
- Optional browser UI consuming the JSON endpoints.
- Static artifact mode for GitHub Actions or local report bundles.
- CORS configuration if a separate local frontend is introduced.
