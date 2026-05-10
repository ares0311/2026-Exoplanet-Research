# Background Search SQLite Schema

## Purpose

Background search automation stores runtime audit state in a top-level SQLite database:

```text
logs/background_search.sqlite3
```

Generated databases are runtime artifacts and should not be committed. Tests should create temporary SQLite databases.

## Schema Version

The current SQLite user version is `2`.

The `schema_migrations` table records migration versions applied by the local code. Future migrations should be additive where possible and must preserve old run ledger records.

## Tables

| Table | Purpose | Append expectation |
| --- | --- | --- |
| `run_ledger` | One row for every scheduled or manual run, including blocked runs. | Append-only. |
| `target_priority_evaluations` | Component priority factors, final score, selected flag, and reason codes for each evaluated target. | Append-only per run. |
| `reviewed_outcomes` | Targets that were reviewed and do not currently require follow-up, plus negative evidence. | Exactly one outcome table row per run, split with `needs_follow_up_outcomes`. |
| `needs_follow_up_outcomes` | Targets requiring follow-up, with trigger reason codes and mandatory tests. | Exactly one outcome table row per run, split with `reviewed_outcomes`. |
| `follow_up_tests` | Deterministic local follow-up test results. | Append-only per run. |
| `draft_reports` | Conservative draft report content and blocking issues. | At most one row per run. |
| `submission_recommendations` | Ranked top-three recommendations, including `do_not_submit_yet` when appropriate. | Append-only per run. |
| `report_exports` | Markdown and HTML export paths generated from draft reports. | Append-only per run. |
| `approval_records` | Human approval state. External submission remains blocked unless explicitly approved. | Append-only per run. |
| `run_locks` | Current non-overlap lock owner. | Runtime coordination table. |
| `schema_migrations` | Applied schema versions. | Append-only by schema version. |

## Core Invariants

- Every run writes exactly one `run_ledger` row.
- Every run writes exactly one primary outcome row across `reviewed_outcomes` and `needs_follow_up_outcomes`.
- Every run ledger row records `config_version` and `config_fingerprint`.
- Child rows must reference an existing `run_ledger.run_id`.
- Reason codes stored in JSON fields must exist in `src/exo_toolkit/background/reason_codes.py`.
- External submission/contact must not be represented as approved unless a human approval record explicitly says so.

## Latest Run Semantics

When CLI commands need a default latest run, they use the run with the newest `completed_at` timestamp, with `started_at` as the tie-breaker. This favors the most recently finished auditable run.

## Validation

Use:

```bash
exo sqlite-integrity
exo validation-summary
```

`sqlite-integrity` checks exactly-one-outcome behavior, config fingerprint presence, orphan child rows, and reason-code validity.
