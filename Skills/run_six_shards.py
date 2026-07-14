"""Launch one repo-native acquisition command as six shards with six workers each.

The child command must be one of the explicitly reviewed shard-capable Skills.
Arguments after ``--`` are forwarded unchanged except that callers may not
override ``--workers``, ``--shard-index``, or ``--shard-count``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARD_COUNT = 6
WORKERS_PER_SHARD = 6
TOTAL_WORKERS = SHARD_COUNT * WORKERS_PER_SHARD
HARD_STORAGE_LIMIT_GB = 100.0
_GB = 1_000_000_000
_RESERVED_FLAGS = ("--workers", "--shard-index", "--shard-count")
_ALLOWED_SCRIPTS = {
    "preflight_tess_asassn_labels.py": "metadata-only exact-TIC ASAS-SN overlap preflight",
    "crossmatch_tess_catalina_labels.py": "bounded TESS-Catalina metadata pilot",
    "process_t1_kepler_batch.py": "Kepler manifest processor (validated at 6x6)",
    "fetch_t1_2_k2_calibration_snippets.py": "native K2 calibration fetcher",
    "star_scanner.py": "prepared-batch live scanner",
}


@dataclass(frozen=True)
class ShardOutcome:
    """Final status for one supervised shard process."""

    shard_index: int
    returncode: int
    log_path: str


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def verify_repo_state() -> None:
    """Fail closed unless this is the authoritative clean main checkout."""
    identity_path = REPO_ROOT / ".agent-project-id"
    identity = identity_path.read_text(encoding="utf-8")
    expected = "EXPECTED_REPO_BASENAME=2026 Exoplanet Research"
    if expected not in identity or REPO_ROOT.name != "2026 Exoplanet Research":
        raise RuntimeError(".agent-project-id does not match the active repository")
    branch = _git_output("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"six-shard downloads must start on main, not {branch!r}")
    status = _git_output("status", "--short")
    if status:
        raise RuntimeError("working tree must be clean before child Run Reports can commit")


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _directories, files in os.walk(path):
        root_path = Path(root)
        for filename in files:
            try:
                total += (root_path / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


def storage_preflight(expected_new_gb: float) -> dict[str, float]:
    """Measure project/cache footprint and enforce the 100 GB hard ceiling."""
    if expected_new_gb < 0:
        raise ValueError("--expected-new-gb cannot be negative")
    managed_paths = (
        REPO_ROOT / "data",
        REPO_ROOT / "datasets",
        REPO_ROOT / "cache",
        REPO_ROOT / ".cache",
        REPO_ROOT / "artifacts",
        REPO_ROOT / "outputs",
        REPO_ROOT / "downloads",
        REPO_ROOT / "tmp",
    )
    repo_managed_bytes = sum(_directory_bytes(path) for path in managed_paths)
    shared_cache = Path.home() / ".lightkurve" / "cache" / "mastDownload"
    shared_cache_bytes = _directory_bytes(shared_cache)
    current_gb = (repo_managed_bytes + shared_cache_bytes) / _GB
    projected_gb = current_gb + expected_new_gb
    free_gb = shutil.disk_usage(REPO_ROOT).free / _GB
    if projected_gb > HARD_STORAGE_LIMIT_GB:
        raise RuntimeError(
            f"projected project-managed data {projected_gb:.2f} GB exceeds "
            f"the {HARD_STORAGE_LIMIT_GB:.0f} GB ceiling"
        )
    return {
        "repo_managed_gb": repo_managed_bytes / _GB,
        "shared_lightkurve_cache_gb": shared_cache_bytes / _GB,
        "current_gb": current_gb,
        "expected_new_gb": expected_new_gb,
        "projected_gb": projected_gb,
        "free_disk_gb": free_gb,
    }


def _validate_forwarded_args(arguments: Sequence[str]) -> tuple[str, ...]:
    forwarded = tuple(arguments)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    for argument in forwarded:
        if any(argument == flag or argument.startswith(f"{flag}=") for flag in _RESERVED_FLAGS):
            raise ValueError(
                f"{argument} is controlled by this launcher; it always uses "
                f"{SHARD_COUNT} shards x {WORKERS_PER_SHARD} workers"
            )
    return forwarded


def build_shard_commands(
    script_name: str,
    forwarded_args: Sequence[str],
    *,
    python_executable: str = sys.executable,
) -> tuple[tuple[str, ...], ...]:
    """Build the six exact child commands after validating the target/flags."""
    if script_name not in _ALLOWED_SCRIPTS:
        allowed = ", ".join(sorted(_ALLOWED_SCRIPTS))
        raise ValueError(f"unsupported --script {script_name!r}; choose one of: {allowed}")
    forwarded = _validate_forwarded_args(forwarded_args)
    script_path = REPO_ROOT / "Skills" / script_name
    if not script_path.is_file():
        raise ValueError(f"reviewed shard script is missing: {script_path}")
    return tuple(
        (
            python_executable,
            str(script_path),
            *forwarded,
            "--workers",
            str(WORKERS_PER_SHARD),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(SHARD_COUNT),
        )
        for shard_index in range(SHARD_COUNT)
    )


def _format_eta(seconds: float) -> str:
    if not seconds or seconds == float("inf"):
        return "unknown"
    return f"{seconds / 60:.0f}m{seconds % 60:.0f}s" if seconds > 90 else f"{seconds:.0f}s"


def _raise_interrupt(_signum: int, _frame: object) -> None:
    """Translate SIGTERM into normal supervisor cleanup."""
    raise KeyboardInterrupt


def supervise_shards(
    commands: Sequence[Sequence[str]],
    log_dir: Path,
    *,
    start_delay_seconds: float,
    heartbeat_seconds: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> tuple[ShardOutcome, ...]:
    """Start all shard children, supervise them, and return ordered outcomes."""
    log_dir.mkdir(parents=True, exist_ok=False)
    lock_path = REPO_ROOT / ".git" / "exo-run-report.lock"
    child_env = os.environ.copy()
    child_env["EXO_RUN_REPORT_LOCK_PATH"] = str(lock_path)
    processes: dict[int, Any] = {}
    log_handles: dict[int, IO[str]] = {}
    log_paths: dict[int, Path] = {}
    completed: dict[int, ShardOutcome] = {}
    started = monotonic_fn()
    next_heartbeat = started + heartbeat_seconds
    try:
        for shard_index, command in enumerate(commands):
            log_path = log_dir / f"shard_{shard_index + 1}_of_{len(commands)}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = popen_factory(
                list(command),
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            processes[shard_index] = process
            log_handles[shard_index] = log_handle
            log_paths[shard_index] = log_path
            print(
                f"  shard {shard_index + 1}/{len(commands)} started pid={process.pid} "
                f"log={log_path}",
                flush=True,
            )
            if shard_index + 1 < len(commands) and start_delay_seconds:
                sleep_fn(start_delay_seconds)

        while len(completed) < len(commands):
            for shard_index, process in processes.items():
                if shard_index in completed:
                    continue
                returncode = process.poll()
                if returncode is None:
                    continue
                log_handles[shard_index].close()
                completed[shard_index] = ShardOutcome(
                    shard_index=shard_index,
                    returncode=int(returncode),
                    log_path=str(log_paths[shard_index]),
                )
                print(
                    f"  shard {shard_index + 1}/{len(commands)} finished "
                    f"status={'PASS' if returncode == 0 else 'FAIL'} code={returncode} "
                    f"log={log_paths[shard_index]}",
                    flush=True,
                )
            now = monotonic_fn()
            if len(completed) < len(commands) and now >= next_heartbeat:
                elapsed = now - started
                rate = len(completed) / elapsed if elapsed else 0.0
                eta = (len(commands) - len(completed)) / rate if rate else float("inf")
                failed = sum(outcome.returncode != 0 for outcome in completed.values())
                print(
                    f"  heartbeat completed={len(completed)}/{len(commands)} "
                    f"active={len(commands) - len(completed)} failed={failed} "
                    f"elapsed={elapsed:.0f}s ETA={_format_eta(eta)}",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            if len(completed) < len(commands):
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
        for open_handle in log_handles.values():
            if not open_handle.closed:
                open_handle.close()
    return tuple(completed[index] for index in range(len(commands)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        choices=sorted(_ALLOWED_SCRIPTS),
        default="process_t1_kepler_batch.py",
    )
    parser.add_argument("--expected-new-gb", type=float, default=1.0)
    parser.add_argument("--start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("forwarded_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.start_delay_seconds < 0 or args.heartbeat_seconds <= 0:
        parser.error("start delay must be non-negative and heartbeat must be positive")

    verify_repo_state()
    storage = storage_preflight(args.expected_new_gb)
    commands = build_shard_commands(args.script, args.forwarded_args)
    print(
        f"Six-shard startup: script={args.script} shards={SHARD_COUNT} "
        f"workers_per_shard={WORKERS_PER_SHARD} total_workers={TOTAL_WORKERS} "
        f"current_data={storage['current_gb']:.2f}GB "
        f"projected={storage['projected_gb']:.2f}GB free={storage['free_disk_gb']:.2f}GB",
        flush=True,
    )
    if args.dry_run:
        for index, command in enumerate(commands, 1):
            print(f"  [{index}/{SHARD_COUNT}] {' '.join(command)}", flush=True)
        print("Dry run COMPLETE: no shard processes started.", flush=True)
        return 0

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = REPO_ROOT / "logs" / "six_shard_runs" / f"{args.script.removesuffix('.py')}_{run_id}"
    outcomes = supervise_shards(
        commands,
        log_dir,
        start_delay_seconds=args.start_delay_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    failures = [outcome for outcome in outcomes if outcome.returncode]
    summary = {
        "script": args.script,
        "shards": SHARD_COUNT,
        "workers_per_shard": WORKERS_PER_SHARD,
        "total_workers": TOTAL_WORKERS,
        "status": "success" if not failures else "failed",
        "outcomes": [outcome.__dict__ for outcome in outcomes],
        "storage_preflight": storage,
    }
    summary_path = log_dir / "launcher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Six-shard COMPLETE: passed={SHARD_COUNT - len(failures)}/{SHARD_COUNT} "
        f"failed={len(failures)} summary={summary_path}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _raise_interrupt)
    raise SystemExit(main())
