#!/usr/bin/env python3
"""Validate retained Phase 2 PTY evidence and update machine-owned state."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_toolkit.prod_state import update_state_from_report

REPO_ROOT = Path(__file__).resolve().parents[1]
PTY_LOG = REPO_ROOT / "logs/prod_closure_evidence/phase2/pty_gate.txt"
PTY_BUNDLE = REPO_ROOT / "artifacts/manifests/exohunter_pty_acceptance.json"
REQUIREMENTS = [
    "LAUNCH-04",
    "CLI-01",
    "CLI-02",
    "CLI-03",
    "UX-CMD-01",
    "UX-CMD-02",
    "UX-CMD-03",
    "UX-IN-01",
    "UX-IN-02",
    "UX-IN-03",
    "UX-IN-04",
    "UX-START-04",
    "UX-TABLE-01",
    "UX-A11Y-01",
]
BOOLEAN_ASSERTIONS = (
    "action_preview_rendered",
    "arrow_navigation_moved_selection",
    "cancel_returns_to_prompt",
    "cursor_hide_show_balanced",
    "escape_closes_palette",
    "guided_fields_shown",
    "help_renders",
    "invalid_input_rejected_inline",
    "palette_describes_parameters",
    "palette_live_filter",
    "palette_opened_without_enter",
    "prompt_after_startup",
    "startup_shows_name_and_version",
    "terminal_canonical_restored",
    "terminal_echo_restored",
)
MATERIAL_PATHS = (
    "docs/HUNTER_PROD_CONTRACT.md",
    "docs/CLI_UX_SPEC.md",
    "scripts/run_pty_gate.sh",
    "tests/test_pty_operator_acceptance.py",
    "artifacts/manifests/exohunter_pty_acceptance.json",
    "logs/prod_closure_evidence/phase2/pty_gate.txt",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _check() -> dict[str, Any]:
    if not PTY_LOG.is_file() or not PTY_BUNDLE.is_file():
        return {
            "check_id": "installed_interactive_pty_operator",
            "requirements": REQUIREMENTS,
            "status": "NOT_EXECUTED",
            "detail": "real-PTY log or behavioural bundle is absent",
        }
    log = PTY_LOG.read_text(encoding="utf-8", errors="replace")
    try:
        bundle = json.loads(PTY_BUNDLE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "check_id": "installed_interactive_pty_operator",
            "requirements": REQUIREMENTS,
            "status": "FAIL",
            "detail": f"PTY behavioural bundle is malformed: {exc}",
        }
    failures: list[str] = []
    if "collected 20 items" not in log or "20 passed" not in log:
        failures.append("pytest denominator is not exactly 20/20 passed")
    if " failed" in log or " skipped" in log or " error" in log.lower():
        failures.append("pytest output contains failed, skipped, or error state")
    for name in BOOLEAN_ASSERTIONS:
        if bundle.get(name) is not True:
            failures.append(f"{name} is not true")
    exact = {
        "bundle_version": "exohunter-pty-acceptance-v1",
        "terminal": "real pty (pty.openpty)",
        "exit_code": 0,
        "non_tty_exit_code": 0,
        "non_tty_ansi_sequences": 0,
        "unrelated_cwd_exit_code": 0,
        "palette_missing_commands": [],
    }
    for key, expected in exact.items():
        if bundle.get(key) != expected:
            failures.append(f"{key}={bundle.get(key)!r}, expected {expected!r}")
    if int(bundle.get("distinct_startup_frames", 0)) < 2:
        failures.append("startup did not contain multiple distinct frames")
    for width in (40, 80, 140):
        if bundle.get(f"overflow_lines_at_{width}") != 0:
            failures.append(f"terminal width {width} overflowed")
    expected_executable = str((REPO_ROOT / ".venv/bin/EXO-Hunter").resolve())
    if str(Path(bundle.get("executable", "")).resolve()) != expected_executable:
        failures.append("bundle executable is not the canonical installed executable")
    return {
        "check_id": "installed_interactive_pty_operator",
        "requirements": REQUIREMENTS,
        "status": "FAIL" if failures else "PASS",
        "detail": (
            "; ".join(failures)
            if failures
            else "20/20 real-PTY keystroke assertions passed against the installed executable"
        ),
    }


def _report(check: dict[str, Any], commit: str) -> dict[str, Any]:
    status = check["status"]
    return {
        "report_version": "exo-hunter-prod-check-v1",
        "contract_version": "HUNTER-PROD-2026-07-30.3",
        "cli_ux_version": "HUNTER-CLI-UX-2026-07-30.3",
        "gate_scope": "PHASE 2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "checks": [check],
        "summary": {
            "total": 1,
            "passed": int(status == "PASS"),
            "failed": int(status == "FAIL"),
            "not_executed": int(status == "NOT_EXECUTED"),
            "passed_denominator": (
                f"{int(status == 'PASS')}/1 checks passed; "
                f"{int(status == 'NOT_EXECUTED')} NOT EXECUTED and excluded from any pass claim"
            ),
        },
        "gate_passed": status == "PASS",
        "prod_ready": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        parser.error(f"--output must be repository-owned: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    report = _report(_check(), commit)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_hashes = {
        relative: _sha256(REPO_ROOT / relative) if (REPO_ROOT / relative).is_file() else "MISSING"
        for relative in MATERIAL_PATHS
    }
    update_state_from_report(
        root=REPO_ROOT,
        report=report,
        phase=2,
        command="bash scripts/run_pty_gate.sh",
        evidence_path=output.relative_to(REPO_ROOT).as_posix(),
        gate_hashes=gate_hashes,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "resolved_executable": str((REPO_ROOT / ".venv/bin/EXO-Hunter").resolve()),
            "terminal": "real pty (pty.openpty)",
        },
    )
    result = report["checks"][0]["status"]
    display = "NOT EXECUTED" if result == "NOT_EXECUTED" else result
    print(f"PRIMARY PHASE GATE: {display}")
    print(report["checks"][0]["detail"])
    print(f"raw evidence: {output.relative_to(REPO_ROOT)}")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
