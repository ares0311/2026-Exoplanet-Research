#!/usr/bin/env python3
"""Frozen primary Phase 3 canonical pipeline acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from exo_toolkit.hunter_ranking import selection_contract
from exo_toolkit.prod_state import update_state_from_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "logs" / "prod_closure_evidence" / "phase3"
BOOTSTRAP = ROOT / "tests" / "hunter_phase3_bootstrap"
EXECUTABLE = ROOT / ".venv" / "bin" / "EXO-Hunter"
CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
REPORT_VERSION = "exo-hunter-prod-check-v1"
NAMED_IDENTITIES = {
    "HIP 60759": "TIC 910001",
    "HIP 61099": "TIC 910002",
    "HIP 3419": "TIC 910003",
}


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


def _run_full_lifecycle(work: Path) -> dict[str, Any]:
    lifecycle_dir = work / "canonical_lifecycle"
    evidence = work / "canonical_lifecycle.json"
    snapshot = work / "canonical_lifecycle.sqlite3.gz"
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "Skills" / "run_hunter_production_acceptance.py"),
        "--work-dir",
        str(lifecycle_dir),
        "--evidence-out",
        str(evidence),
        "--snapshot-out",
        str(snapshot),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (work / "canonical_lifecycle.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (work / "canonical_lifecycle.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"installed canonical lifecycle exited {result.returncode}; "
            f"stderr={result.stderr[-1200:]}"
        )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assertions = payload.get("assertion_results")
    if (
        not isinstance(assertions, dict)
        or len(assertions) != 17
        or not all(value is True for value in assertions.values())
    ):
        raise AssertionError(f"canonical lifecycle assertions were not 17/17: {assertions}")
    commands = payload.get("commands") or []
    if len(commands) != 6 or any(
        Path(str(command_record["argv"][0])).resolve() != EXECUTABLE.resolve()
        for command_record in commands
    ):
        raise AssertionError("not every lifecycle action used the installed executable")
    return {
        "assertions": "17/17",
        "installed_actions": len(commands),
        "evidence": str(evidence.relative_to(ROOT)),
        "stdout": str((work / "canonical_lifecycle.stdout.txt").relative_to(ROOT)),
        "stderr": str((work / "canonical_lifecycle.stderr.txt").relative_to(ROOT)),
    }


def _run_named_identity_case(work: Path) -> dict[str, Any]:
    db = work / "named_identity.sqlite3"
    env = os.environ.copy()
    env["EXO_HUNTER_PHASE3_FIXTURE"] = "1"
    env["PYTHONPATH"] = str(BOOTSTRAP)
    command_text = (
        f"/Create-New-Search --targets 1 --mode new --pool-size 4 --workers 1 "
        f"--db {shlex.quote(str(db))} --json --no-color"
    )
    command = [
        str(EXECUTABLE),
        "--db",
        str(db),
        "--no-animation",
        "--no-color",
        "--command",
        command_text,
    ]
    result = subprocess.run(
        command,
        cwd=Path("/private/tmp"),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "named_identity.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (work / "named_identity.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"installed named-identity command exited {result.returncode}; "
            f"stderr={result.stderr[-1600:]}"
        )

    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        selected = connection.execute(
            "SELECT target_id FROM search_manifest_targets ORDER BY ordinal"
        ).fetchall()
        candidates = {
            row["target_id"]: json.loads(row["candidate_json"])
            for row in connection.execute("SELECT target_id, candidate_json FROM candidate_catalog")
        }
        history_rows = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT identities_json FROM cross_project_search_history"
            )
        ]

    if [row["target_id"] for row in selected] != ["TIC 910004"]:
        raise AssertionError(
            "current sibling history did not exclude HIP 60759, HIP 61099, and "
            f"HIP 3419 before ranking; selected={list(map(dict, selected))}"
        )
    for hip_id, tic_id in NAMED_IDENTITIES.items():
        candidate = candidates.get(tic_id)
        if candidate is None:
            raise AssertionError(f"missing frozen candidate record for {tic_id}")
        if candidate.get("eligible") is not False:
            raise AssertionError(f"{hip_id} remained eligible as New")
        if candidate.get("eligibility_reason") != (f"excluded_cross_project_prior_search:{hip_id}"):
            raise AssertionError(
                f"{hip_id} exclusion lacks exact auditable reason: "
                f"{candidate.get('eligibility_reason')!r}"
            )
        aliases = candidate.get("aliases")
        if aliases != [tic_id.split()[-1], hip_id]:
            raise AssertionError(f"{tic_id} aliases were flattened or malformed: {aliases}")
        provenance = candidate.get("source_provenance")
        if not isinstance(provenance, dict) or provenance.get("selector_version") is None:
            raise AssertionError(f"{tic_id} lacks structured source provenance")
        matching_rows = [identities for identities in history_rows if hip_id in identities]
        if len(matching_rows) != 1 or matching_rows[0] != [hip_id]:
            raise AssertionError(
                f"{hip_id} was not durably imported as one provenance-preserving "
                f"identity record: {matching_rows}"
            )
    return {
        "selected": "TIC 910004",
        "excluded": sorted(NAMED_IDENTITIES),
        "history_rows": len(history_rows),
        "stdout": str((work / "named_identity.stdout.txt").relative_to(ROOT)),
        "stderr": str((work / "named_identity.stderr.txt").relative_to(ROOT)),
    }


def _check_information_gain(work: Path) -> dict[str, Any]:
    db = work / "canonical_lifecycle" / "hunter.sqlite3"
    if selection_contract("follow-up").get("expected_information_gain") != (
        "(1-fpp)*detection_confidence"
    ):
        raise AssertionError("published follow-up information-gain formula drifted")
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        candidates = connection.execute(
            "SELECT c.candidate_json FROM candidate_catalog AS c "
            "JOIN search_manifests AS m USING(search_id) WHERE m.mode='follow-up'"
        ).fetchall()
        registry = connection.execute(
            "SELECT target_id, priority, evidence_json FROM follow_up_registry"
        ).fetchall()
    compared = 0
    for row in candidates:
        candidate = json.loads(row[0])
        expected = candidate["metrics"].get("expected_information_gain")
        if expected is None:
            continue
        matches = [
            entry
            for entry in registry
            if entry["target_id"] == candidate["target_id"]
            and math.isclose(float(entry["priority"]), float(candidate["ranking_score"]))
        ]
        if not matches:
            raise AssertionError(f"no registry evidence for {candidate['target_id']}")
        evidence = json.loads(matches[0]["evidence_json"])
        scores = evidence["scores"]
        calculated = (1.0 - float(scores["false_positive_probability"])) * float(
            scores["detection_confidence"]
        )
        if not math.isclose(float(expected), calculated, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"{candidate['target_id']} persisted information gain {expected} != "
                f"(1-fpp)*confidence {calculated}"
            )
        compared += 1
    if compared < 1:
        raise AssertionError("no follow-up candidate exercised the information-gain formula")
    return {"formula": "(1-fpp)*detection_confidence", "rows_compared": compared}


def _check_tic_request_and_failure() -> dict[str, Any]:
    from Skills import star_scanner

    captured: dict[str, Any] = {}

    class FakeCatalogs:
        @staticmethod
        def query_criteria(**kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

    fake_mast = SimpleNamespace(Catalogs=FakeCatalogs, conf=SimpleNamespace(timeout=0))
    with patch.dict(
        sys.modules,
        {"astroquery": SimpleNamespace(mast=fake_mast), "astroquery.mast": fake_mast},
    ):
        star_scanner._query_tic_criteria_page(3, 37, (6.0, 13.5), query_timeout_seconds=19.0)
    exact = {
        "catalog": "TIC",
        "Tmag": [6.0, 13.5],
        "objType": "STAR",
        "pagesize": 37,
        "page": 3,
    }
    if captured != exact or fake_mast.conf.timeout != 19:
        raise AssertionError(
            f"TIC request drifted: kwargs={captured}, timeout={fake_mast.conf.timeout}"
        )

    calls: list[tuple[int, int, tuple[float, float], float]] = []

    def failing_query(
        page: int,
        pagesize: int,
        tmag_range: tuple[float, float],
        *,
        query_timeout_seconds: float,
    ) -> None:
        calls.append((page, pagesize, tmag_range, query_timeout_seconds))
        raise ConnectionError("phase3 exact failure")

    try:
        star_scanner.select_targets_catalog(
            1,
            tmag_range=(6.0, 13.5),
            pagesize=37,
            retry_attempts=2,
            retry_delay=0.0,
            query_timeout_seconds=19.0,
            query_page_fn=failing_query,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("TIC discovery failure was silently substituted")
    if len(calls) != 2:
        raise AssertionError(f"TIC retry count was not bounded at two: {calls}")
    for fragment in (
        "TIC criteria discovery failed before universe exhaustion",
        "criteria page 1 attempt 1/2: ConnectionError: phase3 exact failure",
        "criteria page 1 attempt 2/2: ConnectionError: phase3 exact failure",
    ):
        if fragment not in message:
            raise AssertionError(f"TIC failure omitted observable detail {fragment!r}: {message}")
    return {"exact_request": exact, "bounded_attempts": len(calls), "failure": message}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/prod_closure_evidence/phase3/canonical_pipeline_gate.json"),
    )
    parser.add_argument("--update-state", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_work"
    work.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    _record(
        checks,
        check_id="phase3-canonical-installed-lifecycle",
        requirements=["PIPE-01", "PIPE-02", "DUR-01", "DUR-02", "DUR-03", "DUR-04"],
        action=lambda: _run_full_lifecycle(work),
    )
    _record(
        checks,
        check_id="phase3-current-federated-identity",
        requirements=["IDENT-01", "IDENT-02", "IDENT-03", "IDENT-04", "PIPE-03"],
        action=lambda: _run_named_identity_case(work),
    )
    _record(
        checks,
        check_id="phase3-exact-information-gain",
        requirements=["RANK-01"],
        action=lambda: _check_information_gain(work),
    )
    _record(
        checks,
        check_id="phase3-exact-tic-request-and-failure",
        requirements=["PIPE-03", "DUR-04"],
        action=_check_tic_request_and_failure,
    )

    commit = _git("rev-parse", "HEAD")
    report = {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "gate_scope": "PHASE 3",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "checks": checks,
        "summary": {
            "passed": sum(check["status"] == "PASS" for check in checks),
            "total": len(checks),
            "result": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_hashes = {
        "docs/HUNTER_PROD_CONTRACT.md": _sha256(ROOT / "docs/HUNTER_PROD_CONTRACT.md"),
        "docs/CLI_UX_SPEC.md": _sha256(ROOT / "docs/CLI_UX_SPEC.md"),
        "scripts/run_canonical_pipeline_gate.py": _sha256(Path(__file__)),
        "tests/hunter_phase3_bootstrap/sitecustomize.py": _sha256(BOOTSTRAP / "sitecustomize.py"),
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
            phase=3,
            command=(
                ".venv/bin/python scripts/run_canonical_pipeline_gate.py "
                f"--output {args.output} --update-state"
            ),
            evidence_path=str(output.relative_to(ROOT)),
            gate_hashes=gate_hashes,
            environment=environment,
        )
    print(
        f"Phase 3 canonical pipeline gate: {report['summary']['result']} "
        f"({report['summary']['passed']}/{report['summary']['total']}); "
        f"evidence={output.relative_to(ROOT)}"
    )
    for check in checks:
        print(f"  {check['status']} {check['check_id']}: {check['observable_result']}")
    return 0 if report["summary"]["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
