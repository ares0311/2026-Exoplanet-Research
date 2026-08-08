#!/usr/bin/env python3
"""Frozen primary Phase 5 real-data New/Follow-up acceptance gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_toolkit.prod_state import update_state_from_report
from exo_toolkit.search_lifecycle import HunterStore

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = ROOT / ".venv" / "bin" / "EXO-Hunter"
CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
REPORT_VERSION = "exo-hunter-prod-check-v1"
MAX_EXECUTION_ATTEMPTS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _latest_state(connection: sqlite3.Connection, search_id: str) -> str:
    row = connection.execute(
        "SELECT state FROM search_state_events WHERE search_id=? ORDER BY id DESC LIMIT 1",
        (search_id,),
    ).fetchone()
    return str(row[0]) if row is not None else "unknown"


def _run_installed(
    *,
    work: Path,
    label: str,
    command_text: str,
    db: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    command = [
        str(EXECUTABLE),
        "--db",
        str(db),
        "--no-animation",
        "--no-color",
        "--command",
        command_text,
    ]
    transcript = work / f"{label}.txt"
    started_at = datetime.now(UTC)
    started = time.monotonic()
    print(f"[{label}] starting installed EXO-Hunter", flush=True)
    with transcript.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=Path("/private/tmp"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(f"[{label}] {line}", end="", flush=True)
        returncode = process.wait()
    completed_at = datetime.now(UTC)
    elapsed = time.monotonic() - started
    print(f"[{label}] exit={returncode} elapsed={elapsed:.1f}s", flush=True)
    return {
        "label": label,
        "argv": command,
        "slash_command": command_text,
        "returncode": returncode,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "transcript": str(transcript.relative_to(ROOT)),
        "transcript_sha256": _sha256(transcript),
    }


def _search(connection: sqlite3.Connection, mode: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM search_manifests WHERE mode=? "
        "AND search_id != 'acceptance-historical-null' ORDER BY created_at DESC LIMIT 1",
        (mode,),
    ).fetchone()
    if row is None:
        raise AssertionError(f"no durable {mode} search was created")
    return row


def _manifest_targets(connection: sqlite3.Connection, search_id: str) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT target_id FROM search_manifest_targets WHERE search_id=? ORDER BY ordinal",
            (search_id,),
        )
    ]


def _execute_until_terminal(
    *,
    work: Path,
    label: str,
    db: Path,
    search_id: str,
    env: dict[str, str],
    commands: list[dict[str, Any]],
) -> None:
    for attempt in range(1, MAX_EXECUTION_ATTEMPTS + 1):
        record = _run_installed(
            work=work,
            label=f"{label}_attempt_{attempt}",
            command_text=(
                f"/Run-New-Search --db {shlex.quote(str(db))} "
                f"--search-id {shlex.quote(search_id)} --workers 6 "
                "--scorer bayesian --pipeline QLP --json --no-color"
            ),
            db=db,
            env=env,
        )
        commands.append(record)
        with sqlite3.connect(db) as connection:
            state = _latest_state(connection, search_id)
        record["durable_state_after"] = state
        if state == "completed":
            return
        if state not in {"partial", "failed"}:
            raise AssertionError(
                f"{label} attempt {attempt} ended in non-resumable state {state!r}"
            )
    raise AssertionError(
        f"{label} did not complete within {MAX_EXECUTION_ATTEMPTS} bounded attempts"
    )


def _candidate_and_result_evidence(
    connection: sqlite3.Connection, search_id: str
) -> dict[str, Any]:
    candidates = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT candidate_json FROM candidate_catalog WHERE search_id=? "
            "ORDER BY canonical_id, target_id",
            (search_id,),
        )
    ]
    histories = [
        {
            **dict(row),
            "result": json.loads(row["result_json"]),
            "provenance": json.loads(row["provenance_json"]),
        }
        for row in connection.execute(
            "SELECT * FROM target_search_history WHERE search_id=? ORDER BY id",
            (search_id,),
        )
    ]
    return {"candidates": candidates, "histories": histories}


def _assert_completed_real_search(
    *,
    db: Path,
    mode: str,
    requested: int,
) -> dict[str, Any]:
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        search = _search(connection, mode)
        search_id = str(search["search_id"])
        targets = _manifest_targets(connection, search_id)
        state = _latest_state(connection, search_id)
        evidence = _candidate_and_result_evidence(connection, search_id)
        config = json.loads(search["config_json"])
        history_identities = {
            identity
            for row in connection.execute(
                "SELECT identities_json FROM cross_project_search_history"
            )
            for identity in json.loads(row[0])
        }
    if len(targets) != requested or search["selected_target_count"] != requested:
        raise AssertionError(
            f"{mode} selected {len(targets)} targets instead of required {requested}"
        )
    if state != "completed":
        raise AssertionError(f"{mode} search is {state!r}, not completed")
    terminal = {
        row["target_id"]
        for row in evidence["histories"]
        if row["status"] in {"candidate_found", "no_signal", "no_data"}
    }
    if terminal != set(targets):
        raise AssertionError(f"{mode} terminal targets {sorted(terminal)} != {targets}")
    for row in evidence["histories"]:
        if not row["provenance"] or not row["result"]:
            raise AssertionError(f"{mode} target lacks durable result/provenance")
    if mode == "new":
        selected_candidates = {
            candidate["target_id"]: candidate
            for candidate in evidence["candidates"]
            if candidate["target_id"] in targets
        }
        for target_id, candidate in selected_candidates.items():
            identities = {
                target_id,
                candidate["canonical_id"],
                *candidate.get("aliases", []),
            }
            normalized = {
                " ".join(str(value).upper().replace("_", " ").split()) for value in identities
            }
            if normalized & history_identities:
                raise AssertionError(
                    f"New target {target_id} intersects prior-search identities: "
                    f"{sorted(normalized & history_identities)}"
                )
        selector_log = config.get("selector_log") or {}
        if not selector_log.get("source_watermarks"):
            raise AssertionError("New search lacks live source watermarks")
        if selector_log.get("remaining_unexplored_universe") not in {0, None}:
            raise AssertionError("New search left an unexplained discovery remainder")
    return {
        "search_id": search_id,
        "manifest_sha256": str(search["manifest_sha256"]),
        "ordered_target_ids": targets,
        "state": state,
        "history_rows": len(evidence["histories"]),
        "source_watermarks": config.get("selector_log", {}).get("source_watermarks"),
        "selection_contract": config.get("selection_contract"),
        "candidates": evidence["candidates"],
        "histories": evidence["histories"],
    }


def _packages() -> dict[str, str]:
    names = (
        "exo-toolkit",
        "astroquery",
        "astropy",
        "lightkurve",
        "numpy",
        "scipy",
        "pydantic",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    return versions


def _write_snapshot(db: Path, destination: Path) -> dict[str, Any]:
    raw = db.read_bytes()
    destination.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "path": str(destination.relative_to(ROOT)),
        "sha256": _sha256(destination),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": destination.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/prod_closure_evidence/phase5/real_data_gate.json"),
    )
    parser.add_argument("--update-state", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_work"
    if work.exists():
        raise SystemExit(f"refusing to reuse real-data evidence directory: {work}")
    work.mkdir(parents=True)
    db = work / "hunter.sqlite3"
    cache = work / "cache"
    (cache / "lightkurve").mkdir(parents=True)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("EXO_HUNTER_ACCEPTANCE_STATE", None)
    env.pop("EXO_HUNTER_PHASE3_FIXTURE", None)
    env["XDG_CACHE_HOME"] = str(cache)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    commands: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    new_evidence: dict[str, Any] | None = None
    follow_up_evidence: dict[str, Any] | None = None

    try:
        create_new = _run_installed(
            work=work,
            label="01_create_new",
            command_text=(
                f"/Create-New-Search --targets 5 --mode new --db {shlex.quote(str(db))} "
                "--workers 6 --json --no-color"
            ),
            db=db,
            env=env,
        )
        commands.append(create_new)
        if create_new["returncode"] != 0:
            raise AssertionError(f"real New creation exited {create_new['returncode']}")
        with sqlite3.connect(db) as connection:
            connection.row_factory = sqlite3.Row
            new_id = str(_search(connection, "new")["search_id"])
        _execute_until_terminal(
            work=work,
            label="02_run_new",
            db=db,
            search_id=new_id,
            env=env,
            commands=commands,
        )
        new_evidence = _assert_completed_real_search(db=db, mode="new", requested=5)
    except Exception as exc:  # noqa: BLE001
        checks.append(
            {
                "check_id": "phase5-real-new",
                "requirements": ["E2E-01", "E2E-04"],
                "status": "FAIL",
                "observable_result": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        checks.append(
            {
                "check_id": "phase5-real-new",
                "requirements": ["E2E-01", "E2E-04"],
                "status": "PASS",
                "observable_result": new_evidence,
            }
        )

    if new_evidence is None:
        checks.append(
            {
                "check_id": "phase5-real-follow-up",
                "requirements": ["E2E-02", "E2E-04"],
                "status": "NOT_EXECUTED",
                "observable_result": "real New prerequisite did not pass",
            }
        )
    else:
        try:
            create_follow_up = _run_installed(
                work=work,
                label="03_create_follow_up",
                command_text=(
                    f"/Create-New-Search --targets 5 --mode follow-up --db "
                    f"{shlex.quote(str(db))} --workers 6 --json --no-color"
                ),
                db=db,
                env=env,
            )
            commands.append(create_follow_up)
            if create_follow_up["returncode"] != 0:
                raise AssertionError(
                    f"real Follow-up creation exited {create_follow_up['returncode']}"
                )
            with sqlite3.connect(db) as connection:
                connection.row_factory = sqlite3.Row
                follow_up_id = str(_search(connection, "follow-up")["search_id"])
            _execute_until_terminal(
                work=work,
                label="04_run_follow_up",
                db=db,
                search_id=follow_up_id,
                env=env,
                commands=commands,
            )
            follow_up_evidence = _assert_completed_real_search(db=db, mode="follow-up", requested=5)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "check_id": "phase5-real-follow-up",
                    "requirements": ["E2E-02", "E2E-04"],
                    "status": "FAIL",
                    "observable_result": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            checks.append(
                {
                    "check_id": "phase5-real-follow-up",
                    "requirements": ["E2E-02", "E2E-04"],
                    "status": "PASS",
                    "observable_result": follow_up_evidence,
                }
            )

    if new_evidence is None or follow_up_evidence is None:
        checks.append(
            {
                "check_id": "phase5-restart-resume",
                "requirements": ["E2E-03", "E2E-04"],
                "status": "NOT_EXECUTED",
                "observable_result": "real New and Follow-up prerequisites did not both pass",
            }
        )
    else:
        try:
            before = {
                "db_sha256": _sha256(db),
                "manifest_checksums": [
                    new_evidence["manifest_sha256"],
                    follow_up_evidence["manifest_sha256"],
                ],
            }
            show = _run_installed(
                work=work,
                label="05_restart_show_follow_ups",
                command_text=(
                    f"/Show-Follow-Ups --db {shlex.quote(str(db))} --status all --json --no-color"
                ),
                db=db,
                env=env,
            )
            commands.append(show)
            if show["returncode"] != 0:
                raise AssertionError(f"restart state read exited {show['returncode']}")
            with sqlite3.connect(db) as connection:
                history_before = connection.execute(
                    "SELECT COUNT(*) FROM target_search_history"
                ).fetchone()[0]
                manifest_before = connection.execute(
                    "SELECT COUNT(*) FROM search_manifests"
                ).fetchone()[0]
            no_repeat = _run_installed(
                work=work,
                label="06_restart_no_repeat",
                command_text=(
                    f"/Run-New-Search --db {shlex.quote(str(db))} --workers 6 "
                    "--scorer bayesian --pipeline QLP --json --no-color"
                ),
                db=db,
                env=env,
            )
            commands.append(no_repeat)
            if no_repeat["returncode"] == 0:
                raise AssertionError("restart repeated completed work as a successful run")
            with sqlite3.connect(db) as connection:
                history_after = connection.execute(
                    "SELECT COUNT(*) FROM target_search_history"
                ).fetchone()[0]
                manifest_after = connection.execute(
                    "SELECT COUNT(*) FROM search_manifests"
                ).fetchone()[0]
            if (history_after, manifest_after) != (history_before, manifest_before):
                raise AssertionError("restart regenerated history or manifests")
            validity = HunterStore(db).validity_summary()
            if not validity["ok"]:
                raise AssertionError(f"restart database validity failed: {validity['issues']}")
            restart_result = {
                "fresh_processes": [show["label"], no_repeat["label"]],
                "history_rows_before_after": [history_before, history_after],
                "manifest_rows_before_after": [manifest_before, manifest_after],
                "manifest_checksums": before["manifest_checksums"],
                "database_validity": "PASS",
                "completed_work_repeated": False,
            }
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "check_id": "phase5-restart-resume",
                    "requirements": ["E2E-03", "E2E-04"],
                    "status": "FAIL",
                    "observable_result": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            checks.append(
                {
                    "check_id": "phase5-restart-resume",
                    "requirements": ["E2E-03", "E2E-04"],
                    "status": "PASS",
                    "observable_result": restart_result,
                }
            )

    snapshot = _write_snapshot(db, work / "hunter.sqlite3.gz") if db.is_file() else None
    evidence_bundle = {
        "commands": commands,
        "database": str(db.relative_to(ROOT)),
        "database_sha256": _sha256(db) if db.is_file() else None,
        "snapshot": snapshot,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "installed_executable": str(EXECUTABLE.resolve()),
            "packages": _packages(),
            "git_head_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "pythonpath_present": "PYTHONPATH" in env,
            "fixture_environment_present": any(
                name in env
                for name in (
                    "EXO_HUNTER_ACCEPTANCE_STATE",
                    "EXO_HUNTER_PHASE3_FIXTURE",
                )
            ),
        },
        "shared_state": {
            "mutation_performed": False,
            "reason": (
                "The gate writes a unique active-repository SQLite database and "
                "evidence bundle; sibling history exports are consumed read-only."
            ),
        },
        "new": new_evidence,
        "follow_up": follow_up_evidence,
    }
    bundle_path = work / "evidence_bundle.json"
    bundle_path.write_text(
        json.dumps(evidence_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_ok = (
        bool(commands)
        and all(Path(ROOT / command["transcript"]).is_file() for command in commands)
        and evidence_bundle["environment"]["pythonpath_present"] is False
        and evidence_bundle["environment"]["fixture_environment_present"] is False
        and snapshot is not None
    )
    checks.append(
        {
            "check_id": "phase5-raw-evidence-bundle",
            "requirements": ["E2E-04"],
            "status": "PASS" if bundle_ok else "FAIL",
            "observable_result": {
                "bundle": str(bundle_path.relative_to(ROOT)),
                "bundle_sha256": _sha256(bundle_path),
                "command_count": len(commands),
                "snapshot": snapshot,
            },
        }
    )

    result = "PASS" if checks and all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "gate_scope": "PHASE 5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "checks": checks,
        "summary": {
            "passed": sum(check["status"] == "PASS" for check in checks),
            "total": len(checks),
            "result": result,
        },
        "evidence_bundle": str(bundle_path.relative_to(ROOT)),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_hashes = {
        "docs/HUNTER_PROD_CONTRACT.md": _sha256(ROOT / "docs/HUNTER_PROD_CONTRACT.md"),
        "scripts/run_real_data_gate.py": _sha256(Path(__file__)),
    }
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "resolved_executable": str(EXECUTABLE.resolve()),
        "data_mode": "live MAST/NEA; no fixtures or mocks",
    }
    if args.update_state:
        update_state_from_report(
            root=ROOT,
            report=report,
            phase=5,
            command=(
                ".venv/bin/python scripts/run_real_data_gate.py "
                f"--output {args.output} --update-state"
            ),
            evidence_path=str(output.relative_to(ROOT)),
            gate_hashes=gate_hashes,
            environment=environment,
        )
    print(
        f"Phase 5 real-data gate: {result} "
        f"({report['summary']['passed']}/{report['summary']['total']}); "
        f"evidence={output.relative_to(ROOT)}",
        flush=True,
    )
    for check in checks:
        print(f"  {check['status']} {check['check_id']}: {check['observable_result']}")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
