#!/usr/bin/env python3
"""PHASE 1 PRIMARY GATE -- installed EXO-Hunter operator surfaces.

The gate builds and executes real operating-system artifacts.  It does not add
``PYTHONPATH``, execute a source file as the product, or use an import as a
substitute for console-script behavior.  Package imports are used only as a
supplemental wheel-completeness assertion after the installed executable has
already been exercised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = "exo-hunter-prod-check-v1"
CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
CLI_UX_VERSION = "HUNTER-CLI-UX-2026-07-30.3"
CANONICAL_SYNC = ("uv", "sync", "--all-extras", "--all-groups")
REQUIRED_COMMANDS = (
    "/New-Search",
    "/Follow-Up-Search",
    "/Run-Search",
    "/Show-Follow-Ups",
    "/Inspect-Target",
    "/Help",
    "/Exit",
)
REQUIRED_ENTRY_POINTS = {
    "EXO-Hunter": "exo_toolkit.hunter_shell:exohunter_entry",
    "Inspect-Target": "exo_toolkit.hunter_cli:inspect_target_entry",
    "prod-check": "exo_toolkit.prod_check:main_entry",
}
REQUIRED_RUNTIME_SKILLS = (
    "star_scanner",
    "run_report",
    "sector_coverage",
    "fetch_jwst_lc",
    "cnn_inference_batcher",
    "cnn_calibrator",
)
MATERIAL_PATHS = (
    "docs/HUNTER_PROD_CONTRACT.md",
    "docs/CLI_UX_SPEC.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/run_installed_operator_gate.py",
)


class Gate:
    """Execute commands, retain raw streams, and derive exact gate states."""

    def __init__(self, *, run_dir: Path, total_steps: int) -> None:
        self.run_dir = run_dir
        self.total_steps = total_steps
        self.started = time.monotonic()
        self.results: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._step = 0

    def _progress(self, label: str, *, complete: bool = False) -> None:
        with self._lock:
            if complete:
                self._step += 1
            elapsed = time.monotonic() - self.started
            rate = self._step / elapsed if self._step and elapsed else 0.0
            remaining = (self.total_steps - self._step) / rate if rate else float("inf")
            eta = "unknown" if remaining == float("inf") else f"{remaining:.0f}s"
            print(
                f"[{self._step}/{self.total_steps}] elapsed={elapsed:.0f}s ETA={eta} {label}",
                flush=True,
            )

    def command(
        self,
        *,
        check_id: str,
        requirements: tuple[str, ...],
        command: Sequence[str],
        cwd: Path,
        environment: dict[str, str],
        expected_codes: frozenset[int] = frozenset({0}),
        assertion: Callable[[subprocess.CompletedProcess[str]], tuple[bool, str]] | None = None,
        timeout: int = 1200,
    ) -> dict[str, Any]:
        """Run one subprocess with separate durable stdout and stderr."""
        self._progress(f"START {check_id}")
        stdout_path = self.run_dir / f"{check_id}.stdout.txt"
        stderr_path = self.run_dir / f"{check_id}.stderr.txt"
        started_at = datetime.now(UTC).isoformat()
        command_started = time.monotonic()
        timed_out = False
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [str(part) for part in command],
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - command_started
                if elapsed > timeout:
                    process.kill()
                    timed_out = True
                    break
                if int(elapsed) > 0 and int(elapsed) % 15 == 0:
                    self._progress(f"RUNNING {check_id} command_elapsed={elapsed:.0f}s")
                time.sleep(1)
            returncode = process.wait()
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        completed = subprocess.CompletedProcess(
            args=[str(part) for part in command],
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )
        code_ok = returncode in expected_codes and not timed_out
        assertion_ok, assertion_detail = (
            assertion(completed) if assertion is not None and code_ok else (code_ok, "exit status")
        )
        status = "PASS" if code_ok and assertion_ok else "FAIL"
        detail = (
            assertion_detail
            if status == "PASS"
            else (
                f"timeout after {timeout}s"
                if timed_out
                else f"exit {returncode}; {assertion_detail}; "
                f"stderr tail={stderr_text[-500:]!r}"
            )
        )
        result = {
            "check_id": check_id,
            "requirements": list(requirements),
            "status": status,
            "detail": detail,
            "command": shlex.join(str(part) for part in command),
            "cwd": str(cwd),
            "exit_code": returncode,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "stdout_path": stdout_path.relative_to(REPO_ROOT).as_posix(),
            "stderr_path": stderr_path.relative_to(REPO_ROOT).as_posix(),
        }
        with self._lock:
            self.results.append(result)
        self._progress(f"{status} {check_id}: {detail}", complete=True)
        return result

    def observation(
        self,
        *,
        check_id: str,
        requirements: tuple[str, ...],
        passed: bool,
        detail: str,
    ) -> dict[str, Any]:
        result = {
            "check_id": check_id,
            "requirements": list(requirements),
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
        with self._lock:
            self.results.append(result)
        self._progress(f"{result['status']} {check_id}: {detail}", complete=True)
        return result


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    environment.update(
        {
            "UV_CACHE_DIR": str(REPO_ROOT / ".uv-cache"),
            "PYTHONNOUSERSITE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    return environment


def _script(run_dir: Path, name: str, content: str) -> Path:
    path = run_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _help_assertion(result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    combined = result.stdout + result.stderr
    missing = [command for command in REQUIRED_COMMANDS if command not in combined]
    ansi = re.search(r"\x1b\[[0-9;?]*[ -/]*[@-~]", combined)
    passed = not missing and ansi is None and "EXO-Hunter" in combined
    return passed, (
        "installed Help and Exit succeeded with all described commands and clean non-TTY output"
        if passed
        else (
            f"missing commands={missing}; ansi={ansi is not None}; "
            f"product_name={'EXO-Hunter' in combined}"
        )
    )


def _invalid_assertion(result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    combined = result.stdout + result.stderr
    sentinel = "Invalid - enter a positive whole number."
    passed = result.returncode != 0 and sentinel in combined and "Traceback" not in combined
    return passed, (
        "invalid scriptable target count returned nonzero with the actionable inline sentinel"
        if passed
        else f"expected nonzero plus {sentinel!r} and no traceback"
    )


def _version_assertion(result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    passed = result.stdout.strip() == "EXO-Hunter 0.5.3"
    return passed, f"installed version output={result.stdout.strip()!r}"


def _wheel_probe_code() -> str:
    return (
        "import importlib.metadata as md, json, pathlib\n"
        "import exo_toolkit\n"
        "from exo_toolkit.hunter_cli import _load_project_skill\n"
        f"required={list(REQUIRED_RUNTIME_SKILLS)!r}\n"
        "skills={name:str(pathlib.Path(_load_project_skill(name).__file__).resolve()) "
        "for name in required}\n"
        "eps={ep.name:ep.value for ep in md.distribution('exo-toolkit').entry_points}\n"
        "package=str(pathlib.Path(exo_toolkit.__file__).resolve())\n"
        "pth=[]\n"
        "for base in map(pathlib.Path, __import__('sys').path):\n"
        "    if base.is_dir():\n"
        "        for item in base.glob('*.pth'):\n"
        "            pth.append({'path':str(item.resolve()),"
        "'text':item.read_text(errors='replace')})\n"
        "print(json.dumps({'package':package,'skills':skills,'entry_points':eps,'pth':pth}, "
        "sort_keys=True))\n"
    )


def _wheel_probe_assertion(
    wheel_env: Path,
) -> Callable[[subprocess.CompletedProcess[str]], tuple[bool, str]]:
    def assert_probe(result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False, "installed package probe did not emit JSON"
        environment_root = wheel_env.resolve()
        package = Path(payload["package"])
        skill_paths = [Path(path) for path in payload["skills"].values()]
        routing_ok = all(
            payload["entry_points"].get(name) == target
            for name, target in REQUIRED_ENTRY_POINTS.items()
        )
        locations_ok = environment_root in package.parents and all(
            environment_root in path.parents for path in skill_paths
        )
        leakage = [
            entry
            for entry in payload["pth"]
            if str(REPO_ROOT.resolve()) in entry["text"] or "/src" in entry["text"]
        ]
        passed = routing_ok and locations_ok and not leakage
        return passed, (
            "wheel contains EXO runtime package and required Skills; console routing is exact; "
            "no PYTHONPATH, editable checkout, source tree, or .pth leakage"
            if passed
            else f"routing_ok={routing_ok}; locations_ok={locations_ok}; leaking_pth={leakage}"
        )

    return assert_probe


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(check["status"] == "PASS" for check in checks)
    failed = sum(check["status"] == "FAIL" for check in checks)
    not_executed = sum(check["status"] == "NOT_EXECUTED" for check in checks)
    return {
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "not_executed": not_executed,
        "passed_denominator": (
            f"{passed}/{len(checks)} checks passed; {not_executed} NOT EXECUTED and "
            "excluded from any pass claim"
        ),
    }


def _hashes() -> dict[str, str]:
    return {
        relative: (
            hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            if (REPO_ROOT / relative).is_file()
            else "MISSING"
        )
        for relative in MATERIAL_PATHS
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _report(checks: list[dict[str, Any]], *, commit: str) -> dict[str, Any]:
    summary = _summary(checks)
    return {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "cli_ux_version": CLI_UX_VERSION,
        "gate_scope": "PHASE 1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "checks": checks,
        "summary": summary,
        "gate_passed": summary["failed"] == 0 and summary["not_executed"] == 0,
        "prod_ready": False,
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _machine_update(
    *,
    output: Path,
    report: dict[str, Any],
    command: str,
    environment: dict[str, str],
    gate_hashes: dict[str, str],
) -> dict[str, Any]:
    try:
        from exo_toolkit.prod_state import update_state_from_report

        update_state_from_report(
            root=REPO_ROOT,
            report=report,
            phase=1,
            command=command,
            evidence_path=output.relative_to(REPO_ROOT).as_posix(),
            gate_hashes=gate_hashes,
            environment=environment,
        )
        ledger = json.loads(
            (REPO_ROOT / "configs" / "HUNTER_PROD_STATE.json").read_text(encoding="utf-8")
        )
        expected = "PASS" if report["gate_passed"] else (
            "FAIL" if report["summary"]["failed"] else "NOT_EXECUTED"
        )
        phase_state = ledger.get("phase_results", {}).get("PHASE 1", {})
        if phase_state.get("gate_result") != expected:
            raise ValueError("machine ledger does not match the Phase 1 report")
    except Exception as exc:  # noqa: BLE001 - state update failure blocks the phase gate
        return {
            "check_id": "state_update",
            "requirements": ["CLAIM-04", "PROD-01"],
            "status": "FAIL",
            "detail": f"deterministic state update failed: {type(exc).__name__}: {exc}",
        }
    return {
        "check_id": "state_update",
        "requirements": ["CLAIM-04", "PROD-01"],
        "status": "PASS",
        "detail": "Phase 1 machine state was derived from and matched to this report",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-state", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        parser.error(f"--output must be inside the active repository: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.parent / (
        "installed_operator_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    )
    run_dir.mkdir()
    environment = _clean_environment()
    sync_env = run_dir / "fresh_sync_env"
    wheel_env = run_dir / "fresh_wheel_env"
    upgrade_env = run_dir / "upgrade_env"
    wheelhouse = run_dir / "wheelhouse"
    wheelhouse.mkdir()
    help_script = _script(run_dir, "help_exit.txt", "/Help\n/Exit\n")
    invalid_command = "/New-Search twenty"
    outside_cwd = Path("/private/tmp").resolve()
    total_steps = 20
    gate = Gate(run_dir=run_dir, total_steps=total_steps)
    print(
        "EXO-Hunter Phase 1 installed-operator gate\n"
        f"total_steps={total_steps} parallel_workers=3 timeout_per_command=1200s\n"
        f"evidence={run_dir.relative_to(REPO_ROOT)}",
        flush=True,
    )

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gate.observation(
        check_id="documented_sync_contract",
        requirements=("LAUNCH-01", "LAUNCH-03"),
        passed="uv sync --all-extras --all-groups" in readme,
        detail="README names the exact canonical operator sync command",
    )
    gate.command(
        check_id="canonical_operator_sync",
        requirements=("LAUNCH-01", "LAUNCH-02", "LAUNCH-03"),
        command=CANONICAL_SYNC,
        cwd=REPO_ROOT,
        environment=environment,
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                gate.command,
                check_id="test_process_surface",
                requirements=("LAUNCH-02",),
                command=(
                    str(REPO_ROOT / ".venv" / "bin" / "pytest"),
                    "tests/test_hunter_shell.py",
                    "-q",
                    "-n0",
                    "-p",
                    "no:cacheprovider",
                ),
                cwd=REPO_ROOT,
                environment=environment,
            ),
            executor.submit(
                gate.command,
                check_id="fresh_synchronized_installation",
                requirements=("LAUNCH-02", "LAUNCH-03"),
                command=CANONICAL_SYNC,
                cwd=REPO_ROOT,
                environment={**environment, "UV_PROJECT_ENVIRONMENT": str(sync_env)},
            ),
            executor.submit(
                gate.command,
                check_id="built_wheel_artifact",
                requirements=("LAUNCH-02", "LAUNCH-03"),
                command=(
                    "uv",
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(wheelhouse),
                    "--no-create-gitignore",
                ),
                cwd=REPO_ROOT,
                environment=environment,
            ),
        ]
        for future in futures:
            future.result()

    wheels = sorted(wheelhouse.glob("exo_toolkit-*.whl"))
    gate.observation(
        check_id="single_built_wheel",
        requirements=("LAUNCH-02",),
        passed=len(wheels) == 1,
        detail=f"expected exactly one built wheel, found {[path.name for path in wheels]}",
    )
    wheel = wheels[0] if len(wheels) == 1 else wheelhouse / "MISSING.whl"

    def install_wheel() -> None:
        gate.command(
            check_id="fresh_wheel_environment",
            requirements=("LAUNCH-02", "LAUNCH-03"),
            command=("uv", "venv", "--python", sys.executable, str(wheel_env)),
            cwd=REPO_ROOT,
            environment=environment,
        )
        gate.command(
            check_id="fresh_wheel_installation",
            requirements=("LAUNCH-02", "LAUNCH-04"),
            command=(
                "uv",
                "pip",
                "install",
                "--strict",
                "--python",
                str(wheel_env),
                str(wheel),
            ),
            cwd=REPO_ROOT,
            environment=environment,
        )

    def install_upgrade_baseline() -> None:
        gate.command(
            check_id="upgrade_environment_baseline",
            requirements=("LAUNCH-02", "LAUNCH-03"),
            command=("uv", "venv", "--python", sys.executable, str(upgrade_env)),
            cwd=REPO_ROOT,
            environment=environment,
        )
        gate.command(
            check_id="upgrade_environment_editable_seed",
            requirements=("LAUNCH-02",),
            command=(
                "uv",
                "pip",
                "install",
                "--python",
                str(upgrade_env),
                "--editable",
                str(REPO_ROOT),
            ),
            cwd=REPO_ROOT,
            environment=environment,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in (executor.submit(install_wheel), executor.submit(install_upgrade_baseline)):
            future.result()

    gate.command(
        check_id="wheel_dependency_consistency",
        requirements=("LAUNCH-02", "LAUNCH-04"),
        command=("uv", "pip", "check", "--python", str(wheel_env)),
        cwd=REPO_ROOT,
        environment=environment,
    )
    gate.command(
        check_id="wheel_runtime_package_completeness",
        requirements=("LAUNCH-04",),
        command=(str(wheel_env / "bin" / "python"), "-c", _wheel_probe_code()),
        cwd=outside_cwd,
        environment=environment,
        assertion=_wheel_probe_assertion(wheel_env),
    )
    gate.command(
        check_id="existing_environment_version",
        requirements=("LAUNCH-01", "LAUNCH-04"),
        command=(str(REPO_ROOT / ".venv" / "bin" / "EXO-Hunter"), "--version"),
        cwd=REPO_ROOT,
        environment=environment,
        assertion=_version_assertion,
    )
    gate.command(
        check_id="existing_environment_help_exit",
        requirements=("LAUNCH-01", "LAUNCH-04"),
        command=(
            str(REPO_ROOT / ".venv" / "bin" / "EXO-Hunter"),
            "--no-color",
            "--no-animation",
            "--script",
            str(help_script),
        ),
        cwd=REPO_ROOT,
        environment=environment,
        assertion=_help_assertion,
    )
    gate.command(
        check_id="fresh_sync_operator_version",
        requirements=("LAUNCH-02", "LAUNCH-04"),
        command=(str(sync_env / "bin" / "EXO-Hunter"), "--version"),
        cwd=REPO_ROOT,
        environment=environment,
        assertion=_version_assertion,
    )
    gate.command(
        check_id="wheel_help_exit_repository_root",
        requirements=("LAUNCH-02", "LAUNCH-04"),
        command=(
            str(wheel_env / "bin" / "EXO-Hunter"),
            "--no-color",
            "--no-animation",
            "--script",
            str(help_script),
        ),
        cwd=REPO_ROOT,
        environment=environment,
        assertion=_help_assertion,
    )
    gate.command(
        check_id="wheel_help_exit_unrelated_directory",
        requirements=("LAUNCH-02", "LAUNCH-04"),
        command=(
            str(wheel_env / "bin" / "EXO-Hunter"),
            "--no-color",
            "--no-animation",
            "--script",
            str(help_script),
        ),
        cwd=outside_cwd,
        environment=environment,
        assertion=_help_assertion,
    )
    gate.command(
        check_id="wheel_invalid_scriptable_input",
        requirements=("LAUNCH-04",),
        command=(
            str(wheel_env / "bin" / "EXO-Hunter"),
            "--no-color",
            "--no-animation",
            "--command",
            invalid_command,
        ),
        cwd=outside_cwd,
        environment=environment,
        expected_codes=frozenset({2}),
        assertion=_invalid_assertion,
    )
    gate.command(
        check_id="upgrade_in_place_to_built_wheel",
        requirements=("LAUNCH-02", "LAUNCH-03"),
        command=(
            "uv",
            "pip",
            "install",
            "--strict",
            "--python",
            str(upgrade_env),
            "--reinstall-package",
            "exo-toolkit",
            str(wheel),
        ),
        cwd=REPO_ROOT,
        environment=environment,
    )
    gate.command(
        check_id="upgraded_operator_version",
        requirements=("LAUNCH-02", "LAUNCH-04"),
        command=(str(upgrade_env / "bin" / "EXO-Hunter"), "--version"),
        cwd=outside_cwd,
        environment=environment,
        assertion=_version_assertion,
    )

    checks = sorted(gate.results, key=lambda check: check["check_id"])
    commit = _git("rev-parse", "HEAD")
    gate_hashes = _hashes()
    report = _report(checks, commit=commit)
    _write_report(output, report)
    command = shlex.join([".venv/bin/python", *sys.argv])
    if args.update_state:
        state_result = _machine_update(
            output=output,
            report=report,
            command=command,
            environment={
                "platform": platform.platform(),
                "python": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "uv": subprocess.run(
                    ["uv", "--version"], capture_output=True, text=True, check=True
                ).stdout.strip(),
                "resolved_executable": str((REPO_ROOT / ".venv/bin/EXO-Hunter").resolve()),
            },
            gate_hashes=gate_hashes,
        )
        checks.append(state_result)
        report = _report(checks, commit=commit)
        _write_report(output, report)

    summary = report["summary"]
    result = "PASS" if report["gate_passed"] else (
        "FAIL" if summary["failed"] else "NOT EXECUTED"
    )
    print(f"PRIMARY PHASE GATE: {result}", flush=True)
    print(summary["passed_denominator"], flush=True)
    print(f"raw evidence: {output.relative_to(REPO_ROOT)}", flush=True)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
