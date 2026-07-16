"""Run the repository quality gates with six pytest shards and six workers each.

The test suite is partitioned by file so every test module belongs to exactly
one shard. Each shard then uses pytest-xdist for in-shard execution. Ruff,
mypy, the incomplete-implementation scan, and the directive-integrity check
run beside the test shards under the same supervising parent process. The
summary JSON records the exact git HEAD SHA and working-tree dirty state this
run verified, per AGENTS.md "No Unsupported Completion Claims".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TEST_SHARD_COUNT = 6
WORKERS_PER_SHARD = 6
TOTAL_TEST_WORKERS = TEST_SHARD_COUNT * WORKERS_PER_SHARD
_TEST_DEFINITION = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.MULTILINE)


@dataclass(frozen=True)
class GateSpec:
    """One independently supervised quality-gate command."""

    name: str
    command: tuple[str, ...]
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class GateOutcome:
    """Final status for one quality-gate child process."""

    name: str
    returncode: int
    elapsed_seconds: float
    log_path: str


def discover_test_files() -> tuple[Path, ...]:
    """Return every repository test module in deterministic order."""
    return tuple(sorted((REPO_ROOT / "tests").rglob("test_*.py")))


def _test_weight(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    return max(1, len(_TEST_DEFINITION.findall(text))), len(text)


def partition_test_files(
    files: Sequence[Path], shard_count: int = TEST_SHARD_COUNT
) -> tuple[tuple[Path, ...], ...]:
    """Greedily balance whole test files by test-definition count and bytes."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if len(files) < shard_count:
        raise ValueError(f"need at least {shard_count} test files, found {len(files)}")
    weighted = sorted(
        ((_test_weight(path), path) for path in files),
        key=lambda item: (item[0][0], item[0][1], str(item[1])),
        reverse=True,
    )
    shards: list[list[Path]] = [[] for _ in range(shard_count)]
    totals = [(0, 0) for _ in range(shard_count)]
    for (test_count, byte_count), path in weighted:
        shard_index = min(range(shard_count), key=lambda index: (*totals[index], index))
        shards[shard_index].append(path)
        previous_tests, previous_bytes = totals[shard_index]
        totals[shard_index] = previous_tests + test_count, previous_bytes + byte_count
    return tuple(tuple(sorted(shard)) for shard in shards)


def build_gate_specs(
    files: Sequence[Path],
    *,
    python_executable: str = str(PYTHON),
    include_static_gates: bool = True,
) -> tuple[GateSpec, ...]:
    """Build static checks plus six disjoint six-worker pytest commands."""
    shards = partition_test_files(files)
    specs: list[GateSpec] = []
    if include_static_gates:
        specs.extend(
            (
                GateSpec("ruff", (python_executable, "-m", "ruff", "check", ".")),
                GateSpec("mypy", (python_executable, "-m", "mypy", "src")),
                GateSpec(
                    "incomplete_implementations",
                    (python_executable, "Skills/check_incomplete_implementations.py"),
                ),
                GateSpec(
                    "directive_integrity",
                    (python_executable, "Skills/check_directive_integrity.py"),
                ),
            )
        )
    # Thirty-six test workers are already the outer CPU parallelism layer.
    # Prevent NumPy/SciPy backends from multiplying that concurrency internally.
    pytest_env = {
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for shard_index, shard_files in enumerate(shards, 1):
        relative_files = tuple(str(path.relative_to(REPO_ROOT)) for path in shard_files)
        specs.append(
            GateSpec(
                name=f"pytest_shard_{shard_index}_of_{TEST_SHARD_COUNT}",
                command=(
                    python_executable,
                    "-m",
                    "pytest",
                    "-n",
                    str(WORKERS_PER_SHARD),
                    "--dist=worksteal",
                    *relative_files,
                ),
                env=pytest_env,
            )
        )
    return tuple(specs)


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Capture the exact repository state this run verified.

    A summary without this is not evidence of *current* correctness once
    the tree changes further — see AGENTS.md "No Unsupported Completion
    Claims". Fails loudly (non-empty "error" key) rather than silently
    omitting the state if git itself is unavailable.
    """
    try:
        head_sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"git_head_sha": head_sha, "git_dirty": bool(porcelain.strip())}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"git_head_sha": None, "git_dirty": None, "git_state_error": str(exc)}


def _format_eta(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    return f"{seconds / 60:.0f}m{seconds % 60:.0f}s" if seconds > 90 else f"{seconds:.0f}s"


def _raise_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def supervise_gates(
    specs: Sequence[GateSpec],
    log_dir: Path,
    *,
    heartbeat_seconds: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> tuple[GateOutcome, ...]:
    """Run all gates concurrently and return outcomes in specification order."""
    log_dir.mkdir(parents=True, exist_ok=False)
    processes: dict[str, Any] = {}
    handles: dict[str, IO[str]] = {}
    paths: dict[str, Path] = {}
    started_at: dict[str, float] = {}
    completed: dict[str, GateOutcome] = {}
    overall_started = monotonic_fn()
    next_heartbeat = overall_started + heartbeat_seconds
    try:
        for spec in specs:
            log_path = log_dir / f"{spec.name}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            child_env = os.environ.copy()
            if spec.env:
                child_env.update(spec.env)
            started_at[spec.name] = monotonic_fn()
            process = popen_factory(
                list(spec.command),
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            processes[spec.name] = process
            handles[spec.name] = log_handle
            paths[spec.name] = log_path
            print(f"  {spec.name} started pid={process.pid} log={log_path}", flush=True)

        while len(completed) < len(specs):
            for spec in specs:
                if spec.name in completed:
                    continue
                process = processes[spec.name]
                returncode = process.poll()
                if returncode is None:
                    continue
                handles[spec.name].close()
                elapsed = monotonic_fn() - started_at[spec.name]
                completed[spec.name] = GateOutcome(
                    name=spec.name,
                    returncode=int(returncode),
                    elapsed_seconds=elapsed,
                    log_path=str(paths[spec.name]),
                )
                print(
                    f"  {spec.name} finished "
                    f"status={'PASS' if returncode == 0 else 'FAIL'} "
                    f"elapsed={elapsed:.1f}s log={paths[spec.name]}",
                    flush=True,
                )
            now = monotonic_fn()
            if len(completed) < len(specs) and now >= next_heartbeat:
                elapsed = now - overall_started
                rate = len(completed) / elapsed if elapsed else 0.0
                eta = (len(specs) - len(completed)) / rate if rate else float("inf")
                failed = sum(outcome.returncode != 0 for outcome in completed.values())
                print(
                    f"  heartbeat completed={len(completed)}/{len(specs)} "
                    f"active={len(specs) - len(completed)} failed={failed} "
                    f"elapsed={elapsed:.0f}s ETA={_format_eta(eta)}",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            if len(completed) < len(specs):
                sleep_fn(min(1.0, heartbeat_seconds))
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for open_handle in handles.values():
            if not open_handle.closed:
                open_handle.close()
    return tuple(completed[spec.name] for spec in specs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0:
        parser.error("heartbeat must be positive")
    if not PYTHON.is_file():
        parser.error(f"repository virtualenv Python is missing: {PYTHON}")

    files = discover_test_files()
    specs = build_gate_specs(files, include_static_gates=not args.tests_only)
    print(
        f"Quality-gate startup: test_files={len(files)} shards={TEST_SHARD_COUNT} "
        f"workers_per_shard={WORKERS_PER_SHARD} "
        f"total_test_workers={TOTAL_TEST_WORKERS} gates={len(specs)}",
        flush=True,
    )
    if args.dry_run:
        for index, spec in enumerate(specs, 1):
            print(f"  [{index}/{len(specs)}] {spec.name}: {' '.join(spec.command)}", flush=True)
        print("Dry run COMPLETE: no quality-gate processes started.", flush=True)
        return 0

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = REPO_ROOT / "logs" / "quality_gates" / run_id
    overall_started = time.monotonic()
    outcomes = supervise_gates(specs, log_dir, heartbeat_seconds=args.heartbeat_seconds)
    elapsed = time.monotonic() - overall_started
    failures = [outcome for outcome in outcomes if outcome.returncode]
    summary = {
        "status": "success" if not failures else "failed",
        "test_files": len(files),
        "test_shards": TEST_SHARD_COUNT,
        "workers_per_shard": WORKERS_PER_SHARD,
        "total_test_workers": TOTAL_TEST_WORKERS,
        "elapsed_seconds": elapsed,
        "outcomes": [outcome.__dict__ for outcome in outcomes],
        **_git_state(REPO_ROOT),
    }
    summary_path = log_dir / "quality_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Quality gates COMPLETE: passed={len(specs) - len(failures)}/{len(specs)} "
        f"failed={len(failures)} elapsed={elapsed:.1f}s summary={summary_path}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _raise_interrupt)
    raise SystemExit(main())
