"""Structured, self-committing completion reports for acquisition/processing scripts.

Every long-running acquisition or processing Skill (fetchers, batch scanners,
corpus builders) should call this module at the end of a successful run to
write a small structured JSON record of what happened, then commit and push
*only that record* to git -- so GitHub reflects real progress without a human
or agent needing to paste console output and manually edit tracking docs.

This is intentionally called a "run report", not a "manifest", to avoid
colliding with this project's existing, distinct use of "manifest" for
leakage-safe training manifests (e.g. ``metadata/t1_1_kepler_training_manifest.jsonl``).

Narrow, explicit exception to the branch/PR policy
---------------------------------------------------
``commit_and_push_report`` pushes directly to whatever branch is checked out
(normally ``main`` on the operator's Mac), bypassing the feature-branch/PR/CI
cycle. This is intentional and narrow: it stages *only* the exact report path
given (never ``git add .`` or ``-A``), so it can never sweep up unrelated
uncommitted work sitting in the operator's working tree. See
``docs/DISCOVERY_RUNBOOK.md`` and ``CLAUDE.md`` for the full policy.

Public API
----------
RunReport -- frozen dataclass describing one run/shard's outcome
report_path_for(script, *, shard_index, shard_count, report_dir) -> Path
append_run_report(report, path) -> None
format_run_report(report) -> str
commit_and_push_report(path, *, message, max_retries, run_fn) -> bool
run_and_commit_report(report, path, *, message, max_retries, run_fn) -> bool
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPORT_DIR = Path("artifacts/manifests/run_reports")


@dataclass(frozen=True)
class RunReport:
    """One acquisition/processing run's (or shard's) completion summary."""

    script: str
    status: str  # "success" | "partial" | "failed"
    started_at: str
    completed_at: str
    elapsed_seconds: float
    items_processed: int
    items_written: int = 0
    items_failed: int = 0
    output_paths: tuple[str, ...] = field(default_factory=tuple)
    shard_index: int | None = None
    shard_count: int | None = None
    items_done_total: int | None = None
    items_total: int | None = None
    percent_done: float | None = None
    notes: str = ""


def report_path_for(
    script: str,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Path:
    """Return the report ledger path for *script*, shard-scoped when applicable.

    Each shard gets its own file (mirroring how sharded scripts already scope
    their output/raw-download paths) so concurrently running shards never
    contend for the same git-tracked file.
    """
    if shard_count is not None and shard_count > 1:
        return report_dir / f"{script}.shard{shard_index}of{shard_count}.jsonl"
    return report_dir / f"{script}.jsonl"


def append_run_report(report: RunReport, path: Path) -> None:
    """Append *report* as one JSON line to *path* (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(report)) + "\n")


def format_run_report(report: RunReport) -> str:
    """Render *report* as a short, human-readable summary (console + PR-style)."""
    lines = [
        f"# Run report: {report.script}",
        "",
        f"- Status: {report.status}",
        f"- Items processed: {report.items_processed}",
        f"- Items written: {report.items_written}",
        f"- Items failed: {report.items_failed}",
        f"- Elapsed: {report.elapsed_seconds:.0f}s",
    ]
    if report.shard_count is not None and report.shard_count > 1:
        lines.append(f"- Shard: {report.shard_index}/{report.shard_count}")
    if report.items_total is not None and report.percent_done is not None:
        lines.append(
            f"- Overall progress: {report.items_done_total}/{report.items_total} "
            f"({report.percent_done:.1f}%)"
        )
    if report.output_paths:
        lines.append(f"- Output: {', '.join(report.output_paths)}")
    if report.notes:
        lines.append(f"- Notes: {report.notes}")
    return "\n".join(lines)


RunFn = Callable[..., subprocess.CompletedProcess[str]]


def _run(run_fn: RunFn, args: list[str]) -> subprocess.CompletedProcess[str]:
    return run_fn(args, capture_output=True, text=True, check=False)


def commit_and_push_report(
    path: Path,
    *,
    message: str,
    max_retries: int = 3,
    run_fn: RunFn = subprocess.run,
) -> bool:
    """Commit and push *only* the file at *path*, retrying on push rejection.

    Never stages anything else (no ``git add .``/``-A``), so unrelated
    uncommitted work in the operator's working tree is never touched. Returns
    ``True`` on success (including the no-op case where the file has no
    changes to commit), ``False`` on any failure -- never raises, so a report
    push failure can never crash the acquisition run that produced real data.
    """
    path = str(path)

    add_result = _run(run_fn, ["git", "add", "--", path])
    if add_result.returncode != 0:
        return False

    commit_result = _run(run_fn, ["git", "commit", "-m", message, "--", path])
    if commit_result.returncode != 0:
        combined = f"{commit_result.stdout}\n{commit_result.stderr}".lower()
        return "nothing to commit" in combined

    branch_result = _run(run_fn, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""

    for _attempt in range(max_retries):
        push_result = _run(run_fn, ["git", "push"])
        if push_result.returncode == 0:
            return True

        if not branch:
            return False
        fetch_result = _run(run_fn, ["git", "fetch", "origin", branch])
        if fetch_result.returncode != 0:
            return False
        rebase_result = _run(run_fn, ["git", "rebase", f"origin/{branch}"])
        if rebase_result.returncode != 0:
            _run(run_fn, ["git", "rebase", "--abort"])
            return False

    return False


def run_and_commit_report(
    report: RunReport,
    path: Path,
    *,
    message: str | None = None,
    max_retries: int = 3,
    run_fn: RunFn = subprocess.run,
) -> bool:
    """Append *report* to *path*, then commit and push it. See :func:`commit_and_push_report`."""
    append_run_report(report, path)
    resolved_message = message or (
        f"Run report: {report.script} ({report.status}, "
        f"{report.items_processed} processed)"
    )
    return commit_and_push_report(
        path, message=resolved_message, max_retries=max_retries, run_fn=run_fn
    )


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_report",
        description="Print the run-report ledger for one script (most recent entries last).",
    )
    parser.add_argument(
        "script", help="Script name whose report ledger to print, e.g. process_t1_kepler_batch"
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    args = parser.parse_args(argv)

    path = report_path_for(
        args.script,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        report_dir=args.report_dir,
    )
    if not path.exists():
        print(f"No run reports yet at {path}")
        return 1
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            print(format_run_report(RunReport(**record)))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
