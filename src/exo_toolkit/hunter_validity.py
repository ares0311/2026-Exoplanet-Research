"""Evidence-derived validity checks for durable EXO-Hunter databases."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from exo_toolkit.hunter_history import load_verified_history_manifest
from exo_toolkit.hunter_models import ExecutionProvenance, HunterCandidate, SearchMode
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_SELECTOR_VERSION,
    NEW_SELECTOR_VERSION,
    OPERATOR_SELECTOR_VERSION,
    selection_contract,
)

REQUIRED_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "candidate_catalog": frozenset(
        {"snapshot_id", "search_id", "target_id", "canonical_id", "candidate_json", "observed_at"}
    ),
    "search_manifests": frozenset(
        {
            "search_id",
            "schema_version",
            "mode",
            "requested_target_count",
            "selected_target_count",
            "candidate_pool_count",
            "selector_version",
            "config_json",
            "manifest_sha256",
            "created_at",
        }
    ),
    "search_manifest_targets": frozenset(
        {
            "search_id",
            "ordinal",
            "snapshot_id",
            "target_id",
            "canonical_id",
            "ranking_score",
            "selection_reason",
            # cross_project_history_validity/_source are deliberately NOT
            # required here: they are nullable columns added later, and a
            # frozen historical snapshot database predating them is still a
            # valid archive. HunterStore.initialize() migrates live databases;
            # requiring them here would retroactively invalidate committed
            # acceptance evidence.
        }
    ),
    "search_state_events": frozenset(
        {"id", "search_id", "state", "detail_json", "created_at"}
    ),
    "search_runs": frozenset(
        {
            "attempt_id",
            "search_id",
            "status",
            "started_at",
            "completed_at",
            "config_json",
            "items_processed",
            "items_succeeded",
            "items_failed",
            "error_message",
        }
    ),
    "target_search_history": frozenset(
        {
            "id",
            "search_id",
            "attempt_id",
            "target_id",
            "status",
            "result_json",
            "provenance_json",
            "error_message",
            "created_at",
        }
    ),
    "cross_project_search_history": frozenset(
        {
            "entry_id",
            "manifest_sha256",
            "manifest_path",
            "source_project",
            "source_search_id",
            "source_path",
            "source_sha256",
            "identities_json",
            "searched_at",
            "source_status",
            "validity_state",
            "provenance_json",
            "imported_at",
        }
    ),
    "follow_up_registry": frozenset(
        {
            "follow_up_id",
            "search_id",
            "target_id",
            "candidate_id",
            "priority",
            "reason",
            "evidence_json",
            "prior_search_provenance_json",
            "recommended_action",
            "search_eligible",
            "revisit_reason",
            "parent_follow_up_id",
            "status",
            "created_at",
        }
    ),
    "follow_up_events": frozenset(
        {
            "id",
            "event_id",
            "follow_up_id",
            "state",
            "related_search_id",
            "detail_json",
            "created_at",
        }
    ),
}

IMMUTABLE_TRIGGER_NAMES = frozenset(
    {
        "hunter_immutable_candidate_catalog_update",
        "hunter_immutable_candidate_catalog_delete",
        "hunter_immutable_search_manifests_update",
        "hunter_immutable_search_manifests_delete",
        "hunter_immutable_manifest_targets_update",
        "hunter_immutable_manifest_targets_delete",
        "hunter_immutable_search_state_update",
        "hunter_immutable_search_state_delete",
        "hunter_immutable_target_history_update",
        "hunter_immutable_target_history_delete",
        "hunter_immutable_cross_project_history_update",
        "hunter_immutable_cross_project_history_delete",
        "hunter_immutable_follow_up_events_update",
        "hunter_immutable_follow_up_events_delete",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_object(raw: Any, *, label: str, issues: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        issues.append(f"{label}: expected a JSON object")
        return {}
    return value


def _parse_time(value: Any, *, label: str, issues: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        issues.append(f"{label}: invalid timestamp: {exc}")
        return None
    if parsed.tzinfo is None:
        issues.append(f"{label}: timestamp has no timezone")
        return None
    return parsed


def _candidate_snapshot_index(
    candidate: HunterCandidate,
    *,
    fallback_index: int,
) -> int:
    pool_ordinal = candidate.source_provenance.get("pool_ordinal")
    if isinstance(pool_ordinal, int) and pool_ordinal > 0:
        return pool_ordinal
    return fallback_index


def _validate_score(
    value: Any,
    *,
    label: str,
    issues: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        issues.append(f"{label}: score is not finite: {value!r}")
    elif not 0.0 <= float(value) <= 1.0:
        issues.append(f"{label}: score is outside [0, 1]: {value!r}")


def _history_sources(
    history_manifest: Path | None,
    *,
    source_root: Path | None,
    issues: list[str],
) -> dict[str, dict[str, Any]]:
    if history_manifest is None:
        return {}
    try:
        payload = load_verified_history_manifest(history_manifest, source_root=source_root)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"history manifest source verification failed: {type(exc).__name__}: {exc}")
        return {}
    return {str(source["search_id"]): dict(source) for source in payload["sources"]}


def validate_hunter_database(
    db_path: Path | str,
    *,
    schema_version: int,
    history_manifest: Path | None = None,
    history_source_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute relational, content, provenance, and immutability evidence.

    ``history_source_root``, when given, resolves every ``source_path`` in
    ``history_manifest`` relative to that exact directory instead of the
    repo-root-walk-up heuristic in
    ``hunter_history._repository_root_for()`` (which finds the nearest
    ancestor containing ``pyproject.toml``+``src/`` and can resolve to the
    wrong root for an external manifest that happens to sit inside some
    other repo's subtree). Pass it explicitly for isolated/scripted
    manifests; leave it ``None`` for the default in-repo manifest.
    """
    path = Path(db_path)
    issues: list[str] = []
    if not path.is_file():
        return {
            "ok": False,
            "issues": [f"Hunter database does not exist: {path}"],
            "schema_version": schema_version,
        }

    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        sqlite_integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if sqlite_integrity != "ok":
            issues.append(f"SQLite integrity_check={sqlite_integrity}")
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            issues.append(f"SQLite foreign_key_check found {len(foreign_key_rows)} violations")
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != schema_version:
            issues.append(f"schema version mismatch: database={user_version} code={schema_version}")

        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table, required_columns in REQUIRED_TABLE_COLUMNS.items():
            if table not in table_names:
                issues.append(f"missing required table: {table}")
                continue
            actual_columns = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            missing_columns = sorted(required_columns - actual_columns)
            if missing_columns:
                issues.append(f"{table}: missing columns {missing_columns}")

        trigger_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        missing_triggers = sorted(IMMUTABLE_TRIGGER_NAMES - trigger_names)
        if missing_triggers:
            issues.append(f"missing immutable-history triggers: {missing_triggers}")

        sources = _history_sources(
            history_manifest, source_root=history_source_root, issues=issues
        )
        manifest_count = 0
        snapshot_count = 0
        manifest_hashes_verified = 0
        snapshot_hashes_verified = 0
        for manifest in connection.execute(
            "SELECT * FROM search_manifests ORDER BY created_at, search_id"
        ):
            manifest_count += 1
            search_id = str(manifest["search_id"])
            config = _json_object(
                manifest["config_json"], label=f"{search_id}.config_json", issues=issues
            )
            catalog_rows = connection.execute(
                "SELECT rowid, * FROM candidate_catalog WHERE search_id=? ORDER BY rowid",
                (search_id,),
            ).fetchall()
            target_rows = connection.execute(
                "SELECT t.*, c.candidate_json, c.search_id AS snapshot_search_id "
                "FROM search_manifest_targets AS t "
                "LEFT JOIN candidate_catalog AS c ON c.snapshot_id=t.snapshot_id "
                "WHERE t.search_id=? ORDER BY t.ordinal",
                (search_id,),
            ).fetchall()
            snapshot_count += len(catalog_rows)
            if len(catalog_rows) != int(manifest["candidate_pool_count"]):
                issues.append(
                    f"{search_id}: candidate pool count {len(catalog_rows)} != "
                    f"{manifest['candidate_pool_count']}"
                )
            if len(target_rows) != int(manifest["selected_target_count"]):
                issues.append(
                    f"{search_id}: selected target count {len(target_rows)} != "
                    f"{manifest['selected_target_count']}"
                )
            if [int(row["ordinal"]) for row in target_rows] != list(
                range(1, len(target_rows) + 1)
            ):
                issues.append(f"{search_id}: manifest ordinals are not contiguous")

            candidate_by_snapshot: dict[str, dict[str, Any]] = {}
            for fallback_index, row in enumerate(catalog_rows, 1):
                label = f"{search_id}.candidate[{fallback_index}]"
                candidate_json = _json_object(
                    row["candidate_json"], label=f"{label}.candidate_json", issues=issues
                )
                try:
                    candidate = HunterCandidate.model_validate(candidate_json)
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"{label}: invalid candidate: {exc}")
                    continue
                expected_index = _candidate_snapshot_index(
                    candidate,
                    fallback_index=fallback_index,
                )
                expected_snapshot = hashlib.sha256(
                    f"{search_id}:{expected_index}:{_canonical_json(candidate_json)}".encode()
                ).hexdigest()
                if expected_snapshot != str(row["snapshot_id"]):
                    issues.append(f"{label}: snapshot SHA-256 mismatch")
                else:
                    snapshot_hashes_verified += 1
                candidate_by_snapshot[str(row["snapshot_id"])] = candidate_json

            selected_candidates: list[dict[str, Any]] = []
            for row in target_rows:
                if row["candidate_json"] is None:
                    issues.append(f"{search_id}: target ordinal {row['ordinal']} is orphaned")
                    continue
                if str(row["snapshot_search_id"]) != search_id:
                    issues.append(f"{search_id}: target references a snapshot from another search")
                selected_candidate_json = candidate_by_snapshot.get(
                    str(row["snapshot_id"])
                )
                if selected_candidate_json is not None:
                    selected_candidates.append(selected_candidate_json)

            selector_version = str(manifest["selector_version"])
            known_selectors = {
                NEW_SELECTOR_VERSION: ("new", False),
                FOLLOW_UP_SELECTOR_VERSION: ("follow-up", False),
                OPERATOR_SELECTOR_VERSION: (str(manifest["mode"]), True),
            }
            if selector_version in known_selectors:
                contract_mode, operator_supplied = known_selectors[selector_version]
                if str(manifest["mode"]) != contract_mode:
                    issues.append(
                        f"{search_id}: selector {selector_version} is invalid for "
                        f"mode={manifest['mode']}"
                    )
                expected_contract = selection_contract(
                    cast(SearchMode, contract_mode),
                    operator_supplied=operator_supplied,
                )
                if config.get("selection_contract") != expected_contract:
                    issues.append(
                        f"{search_id}: selection contract does not match "
                        f"{selector_version}"
                    )
                if not operator_supplied:
                    for candidate_json in candidate_by_snapshot.values():
                        provenance = candidate_json.get("source_provenance", {})
                        if provenance.get("selector_version") != selector_version:
                            issues.append(
                                f"{search_id}: candidate selector provenance does not "
                                f"match {selector_version}"
                            )
                        validity = candidate_json.get("decision_validity")
                        if not isinstance(validity, Mapping):
                            issues.append(
                                f"{search_id}: production candidate lacks "
                                "decision_validity"
                            )
                        elif validity.get("state") not in {
                            "valid",
                            "stale-but-usable",
                        }:
                            issues.append(
                                f"{search_id}: production candidate has unusable "
                                f"decision validity state {validity.get('state')!r}"
                            )
            expected_manifest_hash: str | None = None
            if search_id in sources:
                expected_manifest_hash = _hash_json(sources[search_id])
            elif selector_version == "historical_project_import_v1":
                source_payload = config.get("source_payload")
                if not isinstance(source_payload, Mapping):
                    issues.append(
                        f"{search_id}: historical import lacks a verifiable source payload"
                    )
                else:
                    expected_manifest_hash = _hash_json(source_payload)
                    artifact_path = Path(str(config.get("source_artifact_path", "")))
                    if not artifact_path.is_file():
                        issues.append(
                            f"{search_id}: historical source artifact is missing: "
                            f"{artifact_path}"
                        )
                    else:
                        actual_source_hash = hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest()
                        if actual_source_hash != str(source_payload.get("source_sha256")):
                            issues.append(
                                f"{search_id}: historical source artifact hash mismatch"
                            )
            elif selector_version == "reviewed_prior_import_v1" and selected_candidates:
                history = connection.execute(
                    "SELECT result_json, provenance_json FROM target_search_history "
                    "WHERE search_id=? ORDER BY id LIMIT 1",
                    (search_id,),
                ).fetchone()
                follow_up = connection.execute(
                    "SELECT * FROM follow_up_registry WHERE search_id=? "
                    "ORDER BY created_at LIMIT 1",
                    (search_id,),
                ).fetchone()
                if history is None or follow_up is None:
                    issues.append(f"{search_id}: reviewed import is incomplete")
                else:
                    recommendation = {
                        "candidate_id": follow_up["candidate_id"],
                        "priority": follow_up["priority"],
                        "reason": follow_up["reason"],
                        "evidence": _json_object(
                            follow_up["evidence_json"],
                            label=f"{search_id}.follow_up.evidence",
                            issues=issues,
                        ),
                        "recommended_action": follow_up["recommended_action"],
                        "search_eligible": bool(follow_up["search_eligible"]),
                        "revisit_reason": follow_up["revisit_reason"],
                    }
                    expected_manifest_hash = _hash_json(
                        {
                            "source_search_id": config.get("source_search_id"),
                            "source_attempt_id": config.get("source_attempt_id"),
                            "candidate": selected_candidates[0],
                            "result": _json_object(
                                history["result_json"],
                                label=f"{search_id}.history.result",
                                issues=issues,
                            ),
                            "provenance": _json_object(
                                history["provenance_json"],
                                label=f"{search_id}.history.provenance",
                                issues=issues,
                            ),
                            "recommendation": recommendation,
                        }
                    )
            elif selected_candidates:
                expected_manifest_hash = _hash_json(
                    {
                        "mode": manifest["mode"],
                        "requested": manifest["requested_target_count"],
                        "targets": selected_candidates,
                        "config": config,
                        "selector_version": selector_version,
                    }
                )
            if expected_manifest_hash is None:
                issues.append(f"{search_id}: could not reconstruct manifest SHA-256")
            elif expected_manifest_hash != str(manifest["manifest_sha256"]):
                issues.append(f"{search_id}: manifest SHA-256 mismatch")
            else:
                manifest_hashes_verified += 1

        history_count = 0
        provenance_rows_verified = 0
        status_counts: Counter[str] = Counter()
        for row in connection.execute("SELECT * FROM target_search_history ORDER BY id"):
            history_count += 1
            search_id = str(row["search_id"])
            target_id = str(row["target_id"])
            label = f"{search_id}/{target_id}/history-{row['id']}"
            status = str(row["status"])
            status_counts[status] += 1
            membership = connection.execute(
                "SELECT 1 FROM search_manifest_targets WHERE search_id=? AND target_id=?",
                (search_id, target_id),
            ).fetchone()
            if membership is None:
                issues.append(f"{label}: target is not in exact manifest membership")
            result = _json_object(row["result_json"], label=f"{label}.result", issues=issues)
            provenance = _json_object(
                row["provenance_json"], label=f"{label}.provenance", issues=issues
            )
            if not provenance:
                issues.append(f"{label}: provenance is empty")
            else:
                provenance_rows_verified += 1
            if status == "failed" and not row["error_message"]:
                issues.append(f"{label}: failed result has no error_message")
            historical = result.get("historical_entry", {})
            if isinstance(historical, Mapping):
                _validate_score(
                    historical.get("best_fpp"), label=f"{label}.best_fpp", issues=issues
                )
                _validate_score(
                    historical.get("best_detection_confidence"),
                    label=f"{label}.best_detection_confidence",
                    issues=issues,
                )
            if "schema_version" in provenance:
                try:
                    typed = ExecutionProvenance.model_validate(provenance)
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"{label}: invalid execution provenance: {exc}")
                else:
                    for artifact in typed.model_artifacts:
                        artifact_path = Path(artifact.path)
                        if not artifact_path.is_file():
                            issues.append(f"{label}: model artifact is missing: {artifact.path}")
                            continue
                        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                        if actual != artifact.sha256:
                            issues.append(f"{label}: model artifact hash mismatch: {artifact.path}")

        run_count = 0
        for run in connection.execute("SELECT * FROM search_runs ORDER BY started_at"):
            run_count += 1
            attempt_id = str(run["attempt_id"])
            config = _json_object(
                run["config_json"], label=f"{attempt_id}.config", issues=issues
            )
            rows = connection.execute(
                "SELECT status FROM target_search_history WHERE attempt_id=?",
                (attempt_id,),
            ).fetchall()
            counts = Counter(str(row["status"]) for row in rows)
            processed = sum(counts.values())
            failed = counts["failed"]
            succeeded = processed - failed
            run_counters = (
                int(run["items_processed"]),
                int(run["items_succeeded"]),
                int(run["items_failed"]),
            )
            if run_counters != (processed, succeeded, failed):
                issues.append(
                    f"{attempt_id}: run counters {run_counters} != history "
                    f"{(processed, succeeded, failed)}"
                )
            _parse_time(run["started_at"], label=f"{attempt_id}.started_at", issues=issues)
            if run["completed_at"] is not None:
                _parse_time(
                    run["completed_at"], label=f"{attempt_id}.completed_at", issues=issues
                )
            if config.get("code_version") == "0.3.8":
                for field in ("git_commit", "scorer", "model_artifacts"):
                    if field not in config:
                        issues.append(f"{attempt_id}: v0.3.8 run config lacks {field}")

        follow_up_count = 0
        follow_up_event_count = 0
        for row in connection.execute("SELECT * FROM follow_up_registry ORDER BY created_at"):
            follow_up_count += 1
            follow_up_id = str(row["follow_up_id"])
            evidence = _json_object(
                row["evidence_json"], label=f"{follow_up_id}.evidence", issues=issues
            )
            prior = json.loads(str(row["prior_search_provenance_json"]))
            if not isinstance(prior, list) or not prior:
                issues.append(f"{follow_up_id}: prior-search provenance is empty")
            if not evidence:
                issues.append(f"{follow_up_id}: evidence is empty")
            events = connection.execute(
                "SELECT * FROM follow_up_events WHERE follow_up_id=? ORDER BY id",
                (follow_up_id,),
            ).fetchall()
            follow_up_event_count += len(events)
            dispositions = [
                str(event["state"])
                for event in events
                if event["state"] in {"open", "scheduled", "completed", "deferred"}
            ]
            if not dispositions or dispositions[-1] != str(row["status"]):
                issues.append(f"{follow_up_id}: registry status and lifecycle events disagree")
            if not bool(row["search_eligible"]) and not row["revisit_reason"]:
                issues.append(f"{follow_up_id}: non-executable row lacks revisit_reason")

        durable_counts = {
            "manifests": manifest_count,
            "candidate_snapshots": snapshot_count,
            "manifest_targets": int(
                connection.execute("SELECT COUNT(*) FROM search_manifest_targets").fetchone()[0]
            ),
            "runs": run_count,
            "history_rows": history_count,
            "search_state_events": int(
                connection.execute("SELECT COUNT(*) FROM search_state_events").fetchone()[0]
            ),
            "follow_up_registry": follow_up_count,
            "follow_up_events": follow_up_event_count,
        }
    finally:
        connection.close()

    return {
        "ok": not issues,
        "issues": issues,
        "database_path": str(path),
        "database_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sqlite_integrity": sqlite_integrity,
        "foreign_key_violation_count": len(foreign_key_rows),
        "schema_version": user_version,
        "immutable_trigger_count": len(trigger_names & IMMUTABLE_TRIGGER_NAMES),
        "manifest_hashes_verified": manifest_hashes_verified,
        "snapshot_hashes_verified": snapshot_hashes_verified,
        "provenance_rows_verified": provenance_rows_verified,
        "history_status_counts": dict(sorted(status_counts.items())),
        "counts": durable_counts,
    }
