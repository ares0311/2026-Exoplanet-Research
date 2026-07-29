"""Deterministic clean-state acceptance through the installed EXO-Hunter shell."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from exo_toolkit import __version__  # noqa: E402
from exo_toolkit.hunter_cross_project import DEFAULT_COPIED_HISTORY  # noqa: E402
from exo_toolkit.search_lifecycle import HunterStore  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVIDENCE = (
    REPO_ROOT / "artifacts" / "manifests" / "hunter_live_acceptance_v16.json"
)
DEFAULT_SNAPSHOT = (
    REPO_ROOT / "artifacts" / "evidence" / "hunter_production_snapshot_v16.sqlite3.gz"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic new/follow-up Hunter acceptance through the installed "
            "EXO-Hunter console script and emit a hash-verifiable evidence bundle."
        )
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--snapshot-out", type=Path, default=DEFAULT_SNAPSHOT)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_follow_up_history(work_dir: Path) -> tuple[Path, Path]:
    source_root = work_dir / "history_source"
    source = source_root / "prior-result.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "target_id": "TIC 700001",
                "status": "no_signal",
                "interpretation": "controlled historical null",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = work_dir / "follow_up_history_v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "search_id": "acceptance-historical-null",
                        "mode": "new",
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:01:00+00:00",
                        "searched_by": "EXO-Hunter acceptance",
                        "source_project": "2026 Exoplanet Research",
                        "method_or_data": "controlled historical null",
                        "source_path": source.name,
                        "source_sha256": _sha256(source),
                        "provenance_uri": "acceptance:historical-null",
                        "entries": [
                            {
                                "target_id": "TIC 700001",
                                "canonical_id": "TIC 700001",
                                "status": "no_signal",
                                "searched_at": "2026-01-01T00:00:30+00:00",
                                "ranking_score": 0.1,
                                "metrics": {},
                                "result": {
                                    "composite_interpretation": "no signal",
                                },
                                "best_fpp": None,
                                "best_detection_confidence": None,
                                "best_pathway": None,
                                "error_message": None,
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, source_root


def _run_command(
    *,
    command: Path,
    db: Path,
    slash_command: str,
    work_dir: Path,
    state_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    argv = [
        str(command),
        "--db",
        str(db),
        "--no-animation",
        "--no-color",
        "--command",
        slash_command,
    ]
    started = time.monotonic()
    result = subprocess.run(
        argv,
        cwd=work_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "argv": argv,
        "slash_command": slash_command,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "fixture_state_sha256": _sha256(state_path) if state_path.is_file() else None,
    }


def _database_snapshot(source: Path, target: Path) -> str:
    uncompressed = target.with_suffix("")
    if uncompressed.exists():
        raise RuntimeError(f"refusing to overwrite temporary snapshot: {uncompressed}")
    with (
        sqlite3.connect(source) as source_connection,
        sqlite3.connect(uncompressed) as target_connection,
    ):
        source_connection.backup(target_connection)
    database_sha256 = _sha256(uncompressed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        uncompressed.open("rb") as source_file,
        target.open("wb") as target_file,
        gzip.GzipFile(fileobj=target_file, mode="wb", filename="", mtime=0) as compressed,
    ):
        shutil.copyfileobj(source_file, compressed)
    uncompressed.unlink()
    return database_sha256


def _query_rows(db: Path, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def run_acceptance(
    *,
    work_dir: Path,
    evidence_out: Path,
    snapshot_out: Path,
) -> dict[str, Any]:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"acceptance work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "hunter.sqlite3"
    state_path = work_dir / "fixture_state.json"
    command = REPO_ROOT / ".venv" / "bin" / "EXO-Hunter"
    if not command.is_file():
        raise RuntimeError(f"installed EXO-Hunter command is missing: {command}")
    bootstrap = REPO_ROOT / "tests" / "hunter_acceptance_bootstrap"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(bootstrap), str(REPO_ROOT), str(SRC_ROOT), env.get("PYTHONPATH", ""))
    )
    env["EXO_HUNTER_ACCEPTANCE_STATE"] = str(state_path)
    env["MPLCONFIGDIR"] = str(work_dir / "matplotlib")
    env["CI"] = "1"
    history_manifest, history_source_root = _write_follow_up_history(work_dir)

    commands = [
        (
            "/Create-New-Search --targets 2 --mode new --pool-size 1 --workers 1 "
            f"--cross-project-history-path {shlex.quote(str(DEFAULT_COPIED_HISTORY))} "
            f"--manifest-dir {shlex.quote(str(work_dir / 'manifests'))} "
            "--json --no-color"
        ),
        "/Run-New-Search --workers 1 --scorer bayesian --pipeline QLP --json --no-color",
        "/Run-New-Search --workers 1 --scorer bayesian --pipeline QLP --json --no-color",
        (
            "/Create-New-Search --targets 1 --mode follow-up --workers 1 "
            f"--history-manifest {shlex.quote(str(history_manifest))} "
            f"--history-source-root {shlex.quote(str(history_source_root))} "
            "--json --no-color"
        ),
        "/Run-New-Search --workers 1 --scorer bayesian --pipeline QLP --json --no-color",
        "/Show-Follow-Ups --status all --json --no-color",
    ]
    expected_codes = [0, 2, 0, 0, 0, 0]
    command_results: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, (slash_command, expected) in enumerate(
        zip(commands, expected_codes, strict=True),
        1,
    ):
        result = _run_command(
            command=command,
            db=db,
            slash_command=slash_command,
            work_dir=work_dir,
            state_path=state_path,
            env=env,
        )
        command_results.append(result)
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed > 0 else 0.0
        remaining = (len(commands) - index) / rate if rate > 0 else float("inf")
        print(
            f"[{index}/{len(commands)}] returncode={result['returncode']} "
            f"expected={expected} elapsed={elapsed:.0f}s ETA={remaining:.0f}s",
            flush=True,
        )
        if result["returncode"] != expected:
            raise RuntimeError(
                f"installed command {index} returned {result['returncode']}, "
                f"expected {expected}: {result['stderr']}"
            )

    store = HunterStore(db)
    searches = _query_rows(
        db,
        "SELECT search_id, mode, selector_version, requested_target_count, "
        "selected_target_count, candidate_pool_count, manifest_sha256, created_at "
        "FROM search_manifests "
        "ORDER BY created_at",
    )
    production_searches = [
        row
        for row in searches
        if row["selector_version"] != "historical_project_import_v1"
    ]
    if len(production_searches) != 2:
        raise AssertionError(
            f"expected two production searches, found {len(production_searches)}"
        )
    new_search = store.get_search(str(production_searches[0]["search_id"]))
    follow_up_search = store.get_search(str(production_searches[1]["search_id"]))
    new_manifest_targets = [row["target_id"] for row in new_search["targets"]]
    follow_up_manifest_targets = [
        row["target_id"] for row in follow_up_search["targets"]
    ]
    runs = _query_rows(
        db,
        "SELECT attempt_id, search_id, status, items_processed, items_succeeded, "
        "items_failed, started_at, completed_at FROM search_runs ORDER BY started_at",
    )
    new_runs = [
        row for row in runs if row["search_id"] == new_search["search_id"]
    ]
    target_history = _query_rows(
        db,
        "SELECT id, search_id, attempt_id, target_id, status, result_json, "
        "provenance_json, error_message, created_at FROM target_search_history "
        "ORDER BY id",
    )
    for row in target_history:
        row["result"] = json.loads(row.pop("result_json"))
        row["provenance"] = json.loads(row.pop("provenance_json"))
    follow_ups = store.list_follow_ups(status="all")
    fixture_state = json.loads(state_path.read_text(encoding="utf-8"))
    new_success_targets = {
        row["target_id"]
        for row in target_history
        if row["search_id"] == new_search["search_id"]
        and row["status"] in {"candidate_found", "no_signal", "no_data"}
    }
    follow_up_success_targets = {
        row["target_id"]
        for row in target_history
        if row["search_id"] == follow_up_search["search_id"]
        and row["status"] in {"candidate_found", "no_signal", "no_data"}
    }
    failed_target_attempts = [
        row
        for row in target_history
        if row["search_id"] == new_search["search_id"]
        and row["target_id"] == "TIC 999999"
    ]
    catalog_rows = _query_rows(
        db,
        "SELECT target_id, candidate_json FROM candidate_catalog "
        "WHERE search_id=? ORDER BY target_id",
        (new_search["search_id"],),
    )
    for row in catalog_rows:
        row["candidate"] = json.loads(row.pop("candidate_json"))
        row["eligible"] = row["candidate"]["eligible"]
        row["eligibility_reason"] = row["candidate"]["eligibility_reason"]
        row["ranking_score"] = row["candidate"]["ranking_score"]
    catalog_rows.sort(
        key=lambda row: (-float(row["ranking_score"]), str(row["target_id"]))
    )
    expansion = new_search["config"]["selector_log"][
        "discovery_expansion_attempts"
    ]
    cross_project = new_search["config"]["cross_project_history_import"]
    validity = store.validity_summary(
        history_manifest=history_manifest,
        history_source_root=history_source_root,
    )
    validity_evidence = {
        **validity,
        "history_manifest": str(history_manifest),
        "history_source_root": str(history_source_root),
    }
    strict_bars = [
        recommendation["evidence"].get("meets_strict_production_bar")
        for recommendation in follow_ups
    ]
    assertions = {
        "installed_shell_used_for_every_action": all(
            Path(row["argv"][0]).name == "EXO-Hunter" for row in command_results
        ),
        "fresh_database_started_absent": True,
        "catalog_universe_exhausted": (
            new_search["config"]["selector_log"]["universe_exhausted"] is True
        ),
        "outside_initial_partition_selected": (
            "TIC 999999" in new_manifest_targets
            and int(expansion[0]["retained_limit"]) == 200
            and any(int(row["retained_limit"]) == 201 for row in expansion)
        ),
        "invalid_candidate_excluded": any(
            row["target_id"] == "TIC 800001"
            and row["eligible"] == 0
            and row["eligibility_reason"] == "excluded_known_asassn_variable"
            for row in catalog_rows
        ),
        "invalid_source_record_never_ranked": (
            new_search["config"]["selector_log"]["catalog_rows_seen"] == 202
            and new_search["config"]["selector_log"][
                "raw_candidates_before_exclusion"
            ]
            == 201
            and all(row["target_id"] != "TIC 700002" for row in catalog_rows)
        ),
        "alias_prior_search_excluded": any(
            row["target_id"] == "TIC 800002"
            and str(row["eligibility_reason"]).startswith(
                "excluded_cross_project_prior_search:HIP 74981"
            )
            for row in catalog_rows
        ),
        "stale_copy_policy_explicit": (
            cross_project["validity_state"] == "stale-but-usable"
            and cross_project["source_hashes_verified"] == 0
        ),
        "new_exact_manifest_executed": (
            new_success_targets == set(new_manifest_targets)
        ),
        "follow_up_exact_manifest_executed": (
            follow_up_success_targets == set(follow_up_manifest_targets)
        ),
        "partial_failure_was_loud": (
            new_runs[0]["status"] == "partial"
            and new_runs[0]["items_failed"] == 1
            and command_results[1]["returncode"] == 2
        ),
        "resume_retried_only_failed_target": (
            new_runs[1]["status"] == "completed"
            and new_runs[1]["items_processed"] == 1
            and [row["status"] for row in failed_target_attempts]
            == ["failed", "candidate_found"]
        ),
        "follow_up_uses_prior_evidence": (
            follow_up_manifest_targets
            and follow_up_manifest_targets[0]
            in {row["target_id"] for row in follow_ups}
        ),
        "weak_candidates_remain_ranked": (
            strict_bars and all(value is False for value in strict_bars)
        ),
        "real_scorer_outputs_persisted": all(
            (
                row["status"] != "candidate_found"
                or (
                    row["result"]["individual_scores"]
                    and "scores" in row["result"]["individual_scores"][0]
                    and row["provenance"]["runner"]
                    == "exo_toolkit.cli.run_pipeline"
                )
            )
            for row in target_history
        ),
        "restart_state_valid": bool(validity_evidence["ok"]),
        "ranking_is_reproducible_from_versioned_inputs": (
            new_manifest_targets[0] == "TIC 999999"
            and new_search["selector_version"]
            == new_search["config"]["selection_contract"]["selector_version"]
        ),
    }
    failed_assertions = [name for name, passed in assertions.items() if not passed]
    if failed_assertions:
        raise AssertionError(f"Hunter acceptance assertions failed: {failed_assertions}")

    snapshot_database_sha256 = _database_snapshot(db, snapshot_out)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "acceptance_id": "hunter-production-acceptance-v16",
        "created_at": datetime.now(UTC).isoformat(),
        "toolkit_version": __version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "request": {
            "new": {"requested": 2, "selected": new_manifest_targets},
            "follow_up": {
                "requested": 1,
                "selected": follow_up_manifest_targets,
            },
        },
        "commands": command_results,
        "controlled_inputs": {
            "history_manifest": json.loads(
                history_manifest.read_text(encoding="utf-8")
            ),
            "history_sources": {
                source["source_path"]: (
                    history_source_root / source["source_path"]
                ).read_text(encoding="utf-8")
                for source in json.loads(
                    history_manifest.read_text(encoding="utf-8")
                )["sources"]
            },
        },
        "discovery_coverage": new_search["config"]["selector_log"],
        "validity_report": validity_evidence,
        "provenance_graph": [
            {
                "target_id": row["target_id"],
                "search_id": row["search_id"],
                "attempt_id": row["attempt_id"],
                "source_record": row["provenance"]["candidate_snapshot"][
                    "source_provenance"
                ],
                "decision_validity": row["provenance"]["candidate_snapshot"][
                    "decision_validity"
                ],
                "pipeline_context": row["provenance"]["pipeline_context"],
                "code_version": row["provenance"]["code_version"],
                "git_commit": row["provenance"]["git_commit"],
                "scorer": row["provenance"]["scorer"],
                "runner": row["provenance"]["runner"],
                "result_status": row["status"],
                "result": row["result"],
            }
            for row in target_history
            if "candidate_snapshot" in row["provenance"]
        ],
        "ranking_evidence": {
            "selector_version": new_search["selector_version"],
            "selection_contract": new_search["config"]["selection_contract"],
            "catalog": catalog_rows,
            "follow_up_selector_version": follow_up_search["selector_version"],
            "follow_up_targets": follow_up_search["targets"],
        },
        "selected_targets": {
            "new": new_search["targets"],
            "follow_up": follow_up_search["targets"],
        },
        "search_run": {
            "searches": searches,
            "attempts": runs,
            "target_history": target_history,
            "fixture_state": fixture_state,
        },
        "follow_up_state": follow_ups,
        "assertion_results": assertions,
        "snapshot": {
            "path": _display_path(snapshot_out),
            "database_sha256": snapshot_database_sha256,
            "compressed_sha256": _sha256(snapshot_out),
        },
    }
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Hunter production acceptance passed: {len(assertions)}/{len(assertions)} "
        f"assertions; evidence={evidence_out}; snapshot={snapshot_out}",
        flush=True,
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_acceptance(
            work_dir=args.work_dir,
            evidence_out=args.evidence_out,
            snapshot_out=args.snapshot_out,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Hunter production acceptance failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
