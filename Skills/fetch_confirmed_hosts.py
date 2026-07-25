"""Fetch TIC IDs of confirmed transiting planet hosts from NASA Exoplanet Archive.

Queries the ``ps`` (planetary systems) TAP table for rows where
``tran_flag=1`` (transiting geometry confirmed) and ``tic_id IS NOT NULL``.

Public API
----------
fetch_confirmed_host_tic_ids(*, fetch_fn=None, strict=False) -> frozenset[int]
    Return a frozenset of integer TIC IDs to exclude from discovery scans.
    On any network or parse failure returns an empty frozenset (fails open,
    so the scan continues without confirmed-planet exclusion rather than
    crashing). Strict mode raises for immutable metadata preparation.

This module is primarily consumed as a library (e.g. by
``Skills/star_scanner.py``, which already writes its own Run Report
covering that caller's full scan lifecycle). The CLI entry point below
exists so the confirmed-host exclusion set can also be prepared and
audited as a standalone acquisition step; it always calls the fetcher
with ``strict=True`` so a real success/failure signal reaches the
completion record (AGENTS.md Run Report Policy, Rule 7), independent
of the fail-open default used by in-process callers.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

_NEA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

_QUERY = (
    "SELECT DISTINCT tic_id FROM ps "
    "WHERE tran_flag=1 AND default_flag=1 AND tic_id IS NOT NULL"
)


def _default_fetch(url: str) -> str:
    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def fetch_confirmed_host_tic_ids(
    *,
    fetch_fn: Callable[[str], str] | None = None,
    strict: bool = False,
) -> frozenset[int]:
    """Return TIC IDs of confirmed transiting planet hosts.

    Args:
        fetch_fn: Injectable HTTP fetch callable (accepts URL, returns CSV
            string).  Defaults to ``urllib.request.urlopen``.

    Returns:
        Frozenset of integer TIC IDs. Empty frozenset on any failure unless
        ``strict=True``, which raises so production manifest preparation fails closed.
    """
    _fetch = fetch_fn or _default_fetch

    params = urllib.parse.urlencode({"query": _QUERY, "format": "csv"})
    url = f"{_NEA_TAP_URL}?{params}"

    try:
        raw = _fetch(url)
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()

    try:
        reader = csv.DictReader(io.StringIO(raw))
        tic_ids: set[int] = set()
        for row in reader:
            val = (row.get("tic_id") or "").strip()
            if not val:
                continue
            if val.upper().startswith("TIC "):
                val = val[4:].strip()
            try:
                tic_ids.add(int(float(val)))
            except (ValueError, TypeError):
                continue
        result = frozenset(tic_ids)
        if strict and not result:
            raise RuntimeError("Confirmed-host TAP query returned no TIC IDs")
        return result
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_run_report(
    *,
    started_at: str,
    elapsed_seconds: float,
    status: str,
    n_tic_ids: int,
    output_path: Path,
    notes: str,
    git_run_fn: Any = None,
) -> None:
    """Append and publish one fetch_confirmed_hosts completion report
    (AGENTS.md Rule 7)."""
    report = RunReport(
        script="fetch_confirmed_hosts",
        status=status,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed_seconds,
        items_processed=n_tic_ids,
        items_written=n_tic_ids if status == "success" else 0,
        items_failed=0 if status == "success" else 1,
        output_paths=(str(output_path),),
        notes=notes,
    )
    path = report_path_for("fetch_confirmed_hosts")
    kwargs: dict[str, Any] = {}
    if git_run_fn is not None:
        kwargs["run_fn"] = git_run_fn
    ok = run_and_commit_report(report, path, **kwargs)
    if ok:
        print(f"Run report committed and pushed: {path}", flush=True)
    else:
        print(
            f"Warning: run report written to {path} but commit/push failed",
            flush=True,
        )


def _cli(argv: list[str] | None = None, *, git_run_fn: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_confirmed_hosts",
        description=(
            "Fetch confirmed transiting-planet-host TIC IDs from NASA "
            "Exoplanet Archive as a standalone exclusion-list artifact."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/confirmed_host_tic_ids.json"),
        metavar="JSON",
        help="Output path for sorted TIC ID list (default: data/confirmed_host_tic_ids.json)",
    )
    args = parser.parse_args(argv)

    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    status = "success"
    error_note = ""

    try:
        tic_ids = fetch_confirmed_host_tic_ids(strict=True)
    except Exception as exc:  # noqa: BLE001
        tic_ids = frozenset()
        status = "failed"
        error_note = f" error={exc}"

    sorted_ids = sorted(tic_ids)
    if status == "success":
        # Only write on success: a transient NEA outage during a scheduled
        # re-run must not destroy yesterday's valid exclusion list by
        # overwriting it with an empty one. The Run Report below still
        # records status="failed" either way -- this only protects the
        # actual exclusion-list artifact other tools read from disk.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sorted_ids, indent=2) + "\n")
        print(f"Flag: OK  n_tic_ids={len(sorted_ids)}  output={args.output}")
    else:
        print(
            f"Flag: FETCH_ERROR  n_tic_ids=0  output={args.output} "
            "(NOT written -- preserving any existing file)"
        )

    elapsed = time.monotonic() - start
    _write_run_report(
        started_at=started_at,
        elapsed_seconds=elapsed,
        status=status,
        n_tic_ids=len(sorted_ids),
        output_path=args.output,
        notes=f"strict=True{error_note}",
        git_run_fn=git_run_fn,
    )
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
