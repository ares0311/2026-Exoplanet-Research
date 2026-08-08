"""Deterministic machine-state writer for Hunter production gates.

This module is the only production component authorized to write machine-owned
``VERIFIED`` and ``PROD`` values.  It derives them from a completed executable
gate report, preserves human-authored implementation state as a separate field,
and replaces the ledger atomically only after every input and phase-order
invariant has been validated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
REPORT_VERSION = "exo-hunter-prod-check-v1"
STATE_WRITER = "exo_toolkit.prod_state:update_state_from_report"
VALID_CHECK_RESULTS = frozenset({"PASS", "FAIL", "NOT_EXECUTED"})


def _gate_result(report: dict[str, Any]) -> str:
    """Derive the gate result from checks, never from a claimed summary flag."""
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("gate report must contain at least one executed check")
    statuses: list[str] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or check.get("status") not in VALID_CHECK_RESULTS:
            raise ValueError(f"gate report check {index} has an invalid status")
        statuses.append(str(check["status"]))
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_EXECUTED" in statuses:
        return "NOT_EXECUTED"
    return "PASS"


def _git_identity(root: Path, commit: str) -> dict[str, Any]:
    """Record the tested commit, branch, and dirty state without mutating Git."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {
            "git_head_sha": commit,
            "git_dirty": None,
            "branch": None,
            "identity_source": "gate-report; no Git checkout at writer root",
        }
    if head.stdout.strip() != commit:
        raise ValueError("gate report commit does not match the active Git HEAD")
    branch_result = run("branch", "--show-current")
    status_result = run("status", "--porcelain")
    if branch_result.returncode != 0 or status_result.returncode != 0:
        raise ValueError("unable to determine active Git branch and working-tree state")
    return {
        "git_head_sha": commit,
        "git_dirty": bool(status_result.stdout.strip()),
        "branch": branch_result.stdout.strip(),
    }


def _validate_hashes(gate_hashes: dict[str, str]) -> None:
    if not gate_hashes:
        raise ValueError("gate hashes are required")
    for path, digest in gate_hashes.items():
        if not path or (digest != "MISSING" and not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"invalid material gate hash for {path!r}")


def _require_repo_evidence_path(evidence_path: str) -> None:
    candidate = Path(evidence_path)
    if candidate.is_absolute() or ".." in candidate.parts or not evidence_path.strip():
        raise ValueError("evidence path must be repository-relative")


def _phase_index(value: object) -> int:
    match = re.fullmatch(r"PHASE ([0-7])", str(value))
    if not match:
        raise ValueError(f"ledger must have one exact active phase, got {value!r}")
    return int(match.group(1))


def _require_phase_order(data: dict[str, Any], phase: int) -> None:
    active = _phase_index(data.get("active_phase"))
    previous = data.get("phase_results", {}).get(f"PHASE {phase}", {})
    verified_replay = (
        active == min(phase + 1, 7)
        and previous.get("verification_state") == "VERIFIED"
    )
    if active != phase and not verified_replay:
        raise ValueError(
            f"cannot write PHASE {phase} while the active phase is PHASE {active}"
        )


def _require_report_scope(report: dict[str, Any], phase: int) -> None:
    scope = report.get("gate_scope")
    allowed = {f"PHASE {phase}"}
    if phase == 6:
        allowed.add("FULL PROD")
    if scope not in allowed:
        raise ValueError(f"report scope {scope!r} cannot update PHASE {phase}")


def _require_requirements(
    data: dict[str, Any], report: dict[str, Any]
) -> dict[str, list[str]]:
    requirements = data.get("requirements")
    if not isinstance(requirements, dict) or not requirements:
        raise ValueError("ledger active requirements map is missing")
    statuses: dict[str, list[str]] = {}
    for check in report["checks"]:
        check_requirements = check.get("requirements")
        if not isinstance(check_requirements, (list, tuple)) or not check_requirements:
            raise ValueError(f"check {check.get('check_id')!r} cites no requirements")
        for requirement_id in check_requirements:
            if requirement_id not in requirements:
                raise ValueError(f"gate cites unknown requirement {requirement_id!r}")
            statuses.setdefault(str(requirement_id), []).append(str(check["status"]))
    return statuses


def _requirement_result(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_EXECUTED" in statuses:
        return "NOT_EXECUTED"
    return "PASS"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one repository-owned ledger without exposing a partial file."""
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def update_state_from_report(
    *,
    root: Path,
    report: dict[str, Any],
    phase: int,
    command: str,
    evidence_path: str,
    gate_hashes: dict[str, str],
    environment: dict[str, str],
) -> None:
    """Atomically derive the durable ledger state from one completed gate.

    ``FAIL`` and ``NOT_EXECUTED`` are both nonzero and clear verification.
    ``PROD`` is written only for a passing full-scope Phase 6 report.  Phase
    results may be replayed after a pass so a later regression can revoke the
    prior machine state, but phases may never be skipped.
    """
    if phase not in range(8):
        raise ValueError(f"phase must be in 0..7, got {phase}")
    resolved_root = root.resolve()
    ledger = (resolved_root / "configs" / "HUNTER_PROD_STATE.json").resolve()
    try:
        ledger.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("state ledger resolves outside the active repository") from exc
    if not ledger.is_file():
        raise ValueError("state ledger does not exist")
    if report.get("report_version") != REPORT_VERSION:
        raise ValueError("unsupported gate report version")
    if report.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("gate report contract version does not match")
    commit = report.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{6,64}", commit):
        raise ValueError("gate report lacks a valid tested commit")
    if not command.strip():
        raise ValueError("exact gate command is required")
    if not environment or not all(str(value).strip() for value in environment.values()):
        raise ValueError("complete gate environment identity is required")
    _require_repo_evidence_path(evidence_path)
    _validate_hashes(gate_hashes)
    _require_report_scope(report, phase)

    data = json.loads(ledger.read_text(encoding="utf-8"))
    _require_phase_order(data, phase)
    requirement_statuses = _require_requirements(data, report)
    result = _gate_result(report)
    identity = _git_identity(resolved_root, commit)
    timestamp = str(report.get("generated_at_utc") or "")
    if not timestamp:
        raise ValueError("gate report timestamp is required")
    verification_state = "VERIFIED" if result == "PASS" else None
    gate_exit_code = 0 if result == "PASS" else 1

    data.update(
        {
            "bootstrap_complete": phase > 0 or (phase == 0 and result == "PASS"),
            "prod_status": (
                "PROD"
                if phase == 6 and report.get("gate_scope") == "FULL PROD" and result == "PASS"
                else None
            ),
            "active_phase": (
                f"PHASE {min(phase + 1, 7)}" if result == "PASS" else f"PHASE {phase}"
            ),
            "active_requirement_id": (
                f"PHASE{min(phase + 1, 7)}-PRIMARY-GATE"
                if result == "PASS"
                else next(iter(requirement_statuses))
            ),
            "implementation_state": "IN_PROGRESS" if result == "PASS" else "BLOCKING",
            "gate_execution_state": "EXECUTED",
            "gate_result": result,
            "gate_exit_code": gate_exit_code,
            "gate_command": command,
            "gate_evidence_path": evidence_path,
            "gate_environment": dict(environment),
            "gate_hashes": dict(gate_hashes),
            "tested_code_identity": identity,
            "last_gate_at_utc": timestamp,
        }
    )

    authority = data.setdefault("verification_authority", {})
    authority.update(
        {
            "verified_and_prod_writer": STATE_WRITER,
            "status": "ACTIVE",
            "reason": (
                "Machine verification and PROD state are derived only from executable "
                "gate results and are revoked by FAIL or NOT_EXECUTED."
            ),
        }
    )

    requirement_map = data["requirements"]
    for requirement_id, statuses in requirement_statuses.items():
        requirement = requirement_map[requirement_id]
        requirement_result = _requirement_result(statuses)
        requirement.update(
            {
                "gate_execution_state": "EXECUTED",
                "gate_result": requirement_result,
                "verification_state": (
                    "VERIFIED" if requirement_result == "PASS" else None
                ),
                "evidence_ref": evidence_path,
                "gate_command": command,
                "gate_environment": dict(environment),
                "gate_hashes": dict(gate_hashes),
                "tested_code_identity": identity,
                "tested_at_utc": timestamp,
            }
        )

    phase_key = f"PHASE {phase}"
    phase_results = data.setdefault("phase_results", {})
    phase_results[phase_key] = {
        "implementation_state": "IN_PROGRESS" if result == "PASS" else "BLOCKING",
        "gate_execution_state": "EXECUTED",
        "gate_result": result,
        "gate_exit_code": gate_exit_code,
        "verification_state": verification_state,
        "gate_command": command,
        "environment": dict(environment),
        "observable_result": dict(report["summary"]),
        "raw_evidence_path": evidence_path,
        "gate_hashes": dict(gate_hashes),
        "tested_code_identity": identity,
        "tested_at_utc": timestamp,
    }

    evidence = data.setdefault("evidence", {})
    evidence[f"PHASE{phase}-LATEST"] = {
        "requirement_id": data["active_requirement_id"],
        "phase": phase_key,
        "exact_command": command,
        "environment": dict(environment),
        "observable_assertion": dict(report["summary"]),
        "raw_evidence_path": evidence_path,
        "gate_hashes": dict(gate_hashes),
        "tested_code_identity": identity,
        "tested_at_utc": timestamp,
        "gate_result": result,
    }

    if result == "PASS":
        data["active_blockers"] = [
            blocker
            for blocker in data.get("active_blockers", [])
            if not str(blocker.get("id", "")).startswith(f"PHASE{phase}")
        ]
        data["next_action"] = {
            "requirement_id": data["active_requirement_id"],
            "reason": f"PHASE {phase} passed its deterministic gate; execute the next phase.",
            "requires_user_decision": False,
        }
    else:
        data["active_blockers"] = [
            {
                "id": f"PHASE{phase}-PRIMARY-GATE",
                "priority": "P0",
                "implementation_state": "BLOCKING",
                "gate_result": result,
                "reason": f"The PHASE {phase} primary gate returned {result}.",
                "raw_evidence_path": evidence_path,
            }
        ]
        data["next_action"] = {
            "requirement_id": data["active_requirement_id"],
            "reason": f"Resolve the first PHASE {phase} gate blocker and replay the frozen gate.",
            "requires_user_decision": False,
        }

    _atomic_write_json(ledger, data)
