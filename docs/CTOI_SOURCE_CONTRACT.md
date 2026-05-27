# CTOI SOURCE CONTRACT

## Purpose

Define the stable, fixture-backed contract for using ExoFOP Community TOI
(CTOI) data as an opt-in label source. CTOI data must not be part of the
default training pipeline until a human intentionally runs the fetch and label
assembly steps.

## Provider

- Provider: ExoFOP TESS
- Table: Community TOI download
- URL: `https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv`
- Auth: none expected for the public CSV endpoint
- Rate limit posture: manual or scheduled refresh only; do not query in default
  tests or default validation

## Cache And Fixtures

- Live fetched tables should be cached under `data/`, for example
  `data/exofop_ctoi.csv`.
- Live fetched tables are runtime data and should not be committed unless a
  future decision explicitly promotes a frozen copy as a fixture.
- Default tests use `tests/fixtures/exofop_ctoi_sample.csv` or injected fetch
  functions.

## Accepted Columns

`Skills/fetch_exofop_ctoi.py` accepts common ExoFOP column-name variants for:

| Normalized field | Accepted source columns |
|---|---|
| `toi` | `CTOI`, `ctoi` |
| `tic_id` | `TIC`, `TIC ID`, `tic_id`, `tic id` |
| `disposition` | `User Disposition`, `user_disposition`, `Disposition` |
| `period_days` | `Period (days)`, `period_days`, `Period` |
| `duration_hours` | `Duration (hours)`, `duration_hours`, `Duration (hrs)` |
| `epoch_bjd` | `Epoch (BJD)`, `epoch_bjd`, `Epoch` |
| `n_ratings` | `Num Reports`, `n_ratings`, `Num Ratings`, `num_reports` |

## Label Mapping

The CTOI fetcher normalizes raw dispositions as:

| Raw disposition | Normalized disposition | Training label |
|---|---:|---:|
| `CP` | `cp` | `1` |
| `FP` | `fp` | `0` |
| `EB` | `fp` | `0` |
| `PC` or any other value | `pc` | excluded |

`ctoi_rows_to_label_rows()` emits labels only for normalized `cp` and `fp`
rows. It excludes `pc` rows because unresolved community candidates should not
be treated as supervised labels in the default training flow.

The emitted `confidence` value is a deterministic conflict-resolution weight
derived from `n_ratings`; it is not a calibrated astrophysical probability.

## Guardrails

- CTOI ingestion is opt-in and lives under `Skills/`.
- Default tests must use fixtures or injected fetch functions, never live
  network calls.
- External/community labels can support training data assembly, but they do
  not authorize discovery, confirmation, contact, or submission claims.
- Any future default-training integration must add fixture-based tests and
  update `docs/DATA_SOURCES.md`, `docs/PROJECT_STATUS.md`, and this contract.
