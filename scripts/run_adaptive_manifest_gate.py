#!/usr/bin/env python3
"""Frozen primary Phase 4 adaptive-discovery and manifest lifecycle gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_toolkit.prod_state import update_state_from_report
from exo_toolkit.search_lifecycle import HunterStore

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = ROOT / ".venv" / "bin" / "EXO-Hunter"
CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
REPORT_VERSION = "exo-hunter-prod-check-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _record(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    requirements: list[str],
    action: Callable[[], dict[str, Any]],
) -> None:
    started = time.monotonic()
    try:
        observable = action()
    except Exception as exc:  # noqa: BLE001
        checks.append(
            {
                "check_id": check_id,
                "requirements": requirements,
                "status": "FAIL",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "observable_result": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        checks.append(
            {
                "check_id": check_id,
                "requirements": requirements,
                "status": "PASS",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "observable_result": observable,
            }
        )


def _run_acceptance(work: Path) -> tuple[Path, dict[str, Any]]:
    acceptance_work = work / "acceptance"
    evidence = work / "acceptance.json"
    snapshot = work / "acceptance.sqlite3.gz"
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "Skills" / "run_hunter_production_acceptance.py"),
        "--work-dir",
        str(acceptance_work),
        "--evidence-out",
        str(evidence),
        "--snapshot-out",
        str(snapshot),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (work / "acceptance.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (work / "acceptance.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"installed acceptance exited {result.returncode}; stderr={result.stderr[-1600:]}"
        )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assertions = payload.get("assertion_results")
    if (
        not isinstance(assertions, dict)
        or len(assertions) != 17
        or not all(value is True for value in assertions.values())
    ):
        raise AssertionError(f"underlying acceptance was not 17/17: {assertions}")
    return acceptance_work / "hunter.sqlite3", payload


def _new_search(connection: sqlite3.Connection) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM search_manifests WHERE mode='new' "
        "AND search_id != 'acceptance-historical-null' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise AssertionError("acceptance produced no canonical New search")
    return row


def _check_adaptive_evidence(db: Path) -> dict[str, Any]:
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        search = _new_search(connection)
        config = json.loads(search["config_json"])
        selector_log = config["selector_log"]
        candidate_rows = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT candidate_json FROM candidate_catalog WHERE search_id=?",
                (search["search_id"],),
            )
        ]
        selected = [
            row[0]
            for row in connection.execute(
                "SELECT target_id FROM search_manifest_targets WHERE search_id=? ORDER BY ordinal",
                (search["search_id"],),
            )
        ]

    required_top_level = {
        "requested_target_count",
        "discovered_count",
        "eligible_count",
        "rejection_counts_by_reason",
        "source_identities",
        "source_watermarks",
        "discovery_expansion_attempts",
        "sources_exhausted",
        "remaining_unexplored_universe",
        "termination_reason",
        "quality_distribution",
        "limitations",
    }
    missing = sorted(required_top_level - selector_log.keys())
    if missing:
        raise AssertionError(f"adaptive evidence omits required fields: {missing}")
    if selector_log["requested_target_count"] != search["requested_target_count"]:
        raise AssertionError("requested N disagrees with the durable manifest")
    if selector_log["discovered_count"] != len(candidate_rows):
        raise AssertionError("discovered count disagrees with frozen candidates")
    if selector_log["eligible_count"] != sum(row["eligible"] for row in candidate_rows):
        raise AssertionError("eligible count disagrees with frozen candidates")
    rejection_counts: dict[str, int] = {}
    for candidate in candidate_rows:
        if candidate["eligible"]:
            continue
        reason = str(candidate["eligibility_reason"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    if selector_log["rejection_counts_by_reason"] != dict(sorted(rejection_counts.items())):
        raise AssertionError("rejection counts are not reproducible from frozen candidates")
    if not selector_log["source_identities"] or not selector_log["source_watermarks"]:
        raise AssertionError("source identities or watermarks are empty")
    if selector_log["remaining_unexplored_universe"] != 0:
        raise AssertionError("exhausted adversarial universe reports unexplored candidates")
    if not selector_log["sources_exhausted"]:
        raise AssertionError("source exhaustion is not explicit")
    if selector_log["termination_reason"] not in {
        "top_n_score_upper_bound",
        "accessible_filtered_universe_exhausted",
    }:
        raise AssertionError(
            f"unsupported termination reason: {selector_log['termination_reason']}"
        )
    if not isinstance(selector_log["limitations"], list):
        raise AssertionError("limitations must be an explicit list")

    rounds = selector_log["discovery_expansion_attempts"]
    required_round = {
        "candidates_added",
        "top_n_target_ids",
        "top_n_membership_churn",
        "rank_stability",
        "score_stability",
    }
    if len(rounds) < 2:
        raise AssertionError("adversarial discovery did not expand")
    for index, round_record in enumerate(rounds, 1):
        missing_round = sorted(required_round - round_record.keys())
        if missing_round:
            raise AssertionError(
                f"expansion round {index} omits required evidence: {missing_round}"
            )
    final_top_n = rounds[-1]["top_n_target_ids"]
    if final_top_n != selected:
        raise AssertionError(f"final recorded top-N {final_top_n} != manifest {selected}")
    if "TIC 999999" not in selected:
        raise AssertionError("high-value candidate outside the initial segment was not selected")
    if any("TIC 999999" in row["top_n_target_ids"] for row in rounds[:-1]):
        raise AssertionError("adversarial candidate appeared before the expansion reached it")
    if "TIC 999999" not in rounds[-1]["top_n_membership_churn"]["added"]:
        raise AssertionError("final membership churn does not explain the adversarial addition")

    quality = selector_log["quality_distribution"]
    scores = sorted(float(row["ranking_score"]) for row in candidate_rows if row["eligible"])
    if quality.get("count") != len(scores):
        raise AssertionError("quality distribution count is not reproducible")
    if scores and (
        not abs(float(quality["minimum"]) - scores[0]) < 1e-12
        or not abs(float(quality["maximum"]) - scores[-1]) < 1e-12
    ):
        raise AssertionError("quality distribution bounds are not reproducible")
    return {
        "requested": search["requested_target_count"],
        "discovered": len(candidate_rows),
        "eligible": len(scores),
        "rounds": len(rounds),
        "selected": selected,
        "termination_reason": selector_log["termination_reason"],
    }


def _check_manifest_lifecycle(db: Path, acceptance: dict[str, Any]) -> dict[str, Any]:
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        search = _new_search(connection)
        search_id = str(search["search_id"])
        manifest_targets = [
            row[0]
            for row in connection.execute(
                "SELECT target_id FROM search_manifest_targets WHERE search_id=? ORDER BY ordinal",
                (search_id,),
            )
        ]
        runs = connection.execute(
            "SELECT * FROM search_runs WHERE search_id=? ORDER BY started_at",
            (search_id,),
        ).fetchall()
        histories = connection.execute(
            "SELECT * FROM target_search_history WHERE search_id=? ORDER BY id",
            (search_id,),
        ).fetchall()
        final_state = connection.execute(
            "SELECT state FROM search_state_events WHERE search_id=? ORDER BY id DESC LIMIT 1",
            (search_id,),
        ).fetchone()[0]

    validity = HunterStore(db).validity_summary()
    if not validity["ok"]:
        raise AssertionError(f"durable manifest validity failed: {validity['issues']}")
    if len(runs) != 2 or [row["status"] for row in runs] != ["partial", "completed"]:
        raise AssertionError("partial/resume lifecycle was not durable and explicit")
    first_attempt = [row for row in histories if row["attempt_id"] == runs[0]["attempt_id"]]
    second_attempt = [row for row in histories if row["attempt_id"] == runs[1]["attempt_id"]]
    if [row["target_id"] for row in first_attempt] != manifest_targets:
        raise AssertionError("first execution did not preserve exact manifest ordering")
    failed_first = [row["target_id"] for row in first_attempt if row["status"] == "failed"]
    if len(failed_first) != 1 or [row["target_id"] for row in second_attempt] != failed_first:
        raise AssertionError("resume regenerated, substituted, or retried completed targets")
    failed_row = next(row for row in first_attempt if row["status"] == "failed")
    if not failed_row["error_message"] or not failed_row["provenance_json"]:
        raise AssertionError("per-target failure or provenance was not durable")
    if final_state != "completed":
        raise AssertionError(f"resumed search final state is {final_state!r}")
    create_command = acceptance["commands"][0]
    if '"manifest_csv": null' not in create_command["stdout"]:
        raise AssertionError("small search did not prove SQLite execution without CSV")
    if search["manifest_sha256"] not in create_command["stdout"]:
        raise AssertionError("created checksum differs from the durable executed manifest")
    return {
        "search_id": search_id,
        "manifest_sha256": search["manifest_sha256"],
        "ordered_target_ids": manifest_targets,
        "attempt_statuses": [row["status"] for row in runs],
        "retried_target_ids": failed_first,
        "final_state": final_state,
        "csv_required_for_execution": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/prod_closure_evidence/phase4/adaptive_manifest_gate.json"),
    )
    parser.add_argument("--update-state", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_work"
    work.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    db: Path | None = None
    acceptance: dict[str, Any] | None = None
    try:
        db, acceptance = _run_acceptance(work)
    except Exception as exc:  # noqa: BLE001
        setup_error = f"{type(exc).__name__}: {exc}"
    else:
        setup_error = None

    def require_acceptance() -> tuple[Path, dict[str, Any]]:
        if setup_error is not None or db is None or acceptance is None:
            raise RuntimeError(f"acceptance setup failed: {setup_error}")
        return db, acceptance

    _record(
        checks,
        check_id="phase4-adaptive-adversarial-discovery",
        requirements=["DISC-01", "DISC-02", "DISC-03", "RANK-01"],
        action=lambda: _check_adaptive_evidence(require_acceptance()[0]),
    )
    _record(
        checks,
        check_id="phase4-frozen-manifest-resume-persistence",
        requirements=["DUR-01", "DUR-02", "DUR-03", "DUR-04"],
        action=lambda: _check_manifest_lifecycle(*require_acceptance()),
    )

    report = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "gate_scope": "PHASE 4",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "checks": checks,
        "summary": {
            "passed": sum(check["status"] == "PASS" for check in checks),
            "total": len(checks),
            "result": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        },
        "raw_evidence": {
            "acceptance_stdout": str((work / "acceptance.stdout.txt").relative_to(ROOT)),
            "acceptance_stderr": str((work / "acceptance.stderr.txt").relative_to(ROOT)),
            "acceptance_bundle": str((work / "acceptance.json").relative_to(ROOT)),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_hashes = {
        "docs/HUNTER_PROD_CONTRACT.md": _sha256(ROOT / "docs/HUNTER_PROD_CONTRACT.md"),
        "scripts/run_adaptive_manifest_gate.py": _sha256(Path(__file__)),
        "Skills/run_hunter_production_acceptance.py": _sha256(
            ROOT / "Skills/run_hunter_production_acceptance.py"
        ),
        "tests/hunter_acceptance_bootstrap/sitecustomize.py": _sha256(
            ROOT / "tests/hunter_acceptance_bootstrap/sitecustomize.py"
        ),
    }
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "resolved_executable": str(EXECUTABLE.resolve()),
        "terminal": "non-interactive installed console-script subprocesses",
    }
    if args.update_state:
        update_state_from_report(
            root=ROOT,
            report=report,
            phase=4,
            command=(
                ".venv/bin/python scripts/run_adaptive_manifest_gate.py "
                f"--output {args.output} --update-state"
            ),
            evidence_path=str(output.relative_to(ROOT)),
            gate_hashes=gate_hashes,
            environment=environment,
        )
    print(
        f"Phase 4 adaptive manifest gate: {report['summary']['result']} "
        f"({report['summary']['passed']}/{report['summary']['total']}); "
        f"evidence={output.relative_to(ROOT)}"
    )
    for check in checks:
        print(f"  {check['status']} {check['check_id']}: {check['observable_result']}")
    return 0 if report["summary"]["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
