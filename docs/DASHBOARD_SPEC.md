# Candidate Dashboard Specification

## Purpose

Define the local dashboard surface for reviewing exoplanet candidate signals,
follow-up targets, and background-search outputs.

The dashboard is a review aid. It must not claim discovery, confirmation, or
external validation. It should make false-positive evidence, negative evidence,
missing evidence, and blocking issues visible next to any positive signal
evidence.

## Initial Implementation

The first implementation is a static HTML exporter:

```bash
python Skills/candidate_dashboard_export.py results.json --output reports/dashboard.html
```

This avoids new server dependencies and keeps the workflow compatible with
offline local review, GitHub artifacts, email attachments, and Dropbox-backed
project folders.

## Supported Inputs

The exporter accepts existing local JSON shapes:

- `exo --output` candidate rows
- `batch_scan.py` result rows
- ranked candidate rows from `rank_candidates.py`
- wrapped JSON with top-level `candidates` or `rows`
- candidate-database style rows after JSON export or conversion

Live catalog access is out of scope. Source paths are recorded in loaded rows
for provenance.

## Required Views

The static dashboard should include:

- overview counts
- minimum and median false-positive probability
- blocked count
- sortable or scannable candidate table
- per-candidate detail sections
- positive evidence
- false-positive and negative evidence
- blocking issues
- pathway or status labels
- source provenance when available

## Guardrails

- Use "candidate signal" or "follow-up target" language.
- Do not use "confirmed planet" as a candidate status or heading.
- Preserve false-positive probability when present.
- Render missing scores as unknown or `n/a`, not as zero.
- Treat blocking issues as a first-class risk band.
- Do not query live services in default operation or tests.
- Do not write generated dashboards into tracked runtime paths by default.

## Non-Goals

- No hosted web service in the first implementation.
- No live MAST, ExoFOP, Gaia, or NASA Exoplanet Archive queries.
- No external submission workflow.
- No authentication or multi-user state.
- No replacement for the SQLite background automation ledger.

## Future Extensions

- Local read-only API for candidate browsing. Implemented by
  `Skills/candidate_api.py`; see `docs/API_SPEC.md`.
- Integration with `logs/background_search.sqlite3` summaries.
- Embedded phase-fold plots when plot artifacts are supplied.
- Browser UI once the static data contract is stable. Implemented by
  `Skills/candidate_browser_ui.py`.
