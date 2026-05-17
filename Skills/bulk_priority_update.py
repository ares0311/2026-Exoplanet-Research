"""Re-score an existing scan_log.json with a new priority function.

Reads the scan log produced by star_scanner.py, recomputes priority_score for
every entry using the supplied priority_fn, and writes the updated log
atomically.

Public API
----------
update_priorities(log_path, *, priority_fn, output_path) -> dict
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _default_priority_fn(entry: dict[str, Any]) -> float:
    """Reproduce the star_scanner.priority_score heuristic from entry fields."""
    try:
        from Skills.star_scanner import priority_score  # noqa: PLC0415
        return priority_score(
            entry.get("tmag"),
            entry.get("teff"),
            entry.get("n_sectors", 0),
            entry.get("contratio"),
        )
    except ImportError:
        return float(entry.get("priority_score", 0.5))


def update_priorities(
    log_path: Path | str,
    *,
    priority_fn: Callable[[dict[str, Any]], float] | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Re-score all entries in a scan log with a new priority function.

    Args:
        log_path: Path to the existing ``scan_log.json``.
        priority_fn: Callable ``(entry_dict) -> float`` that returns a new
            priority score in [0, 1].  Defaults to the built-in
            :func:`star_scanner.priority_score` heuristic.
        output_path: Where to write the updated log.  Defaults to
            ``log_path`` (in-place update).

    Returns:
        Summary dict with ``n_entries``, ``n_updated``, ``mean_old_score``,
        ``mean_new_score``.
    """
    log_path = Path(log_path)
    if output_path is None:
        output_path = log_path

    data: dict[str, Any] = json.loads(log_path.read_text())
    entries: dict[str, Any] = data.get("entries", {})

    _fn = priority_fn if priority_fn is not None else _default_priority_fn

    old_scores: list[float] = []
    new_scores: list[float] = []
    n_updated = 0

    for entry in entries.values():
        old = float(entry.get("priority_score", 0.0))
        new = float(_fn(entry))
        old_scores.append(old)
        new_scores.append(new)
        if abs(new - old) > 1e-9:
            n_updated += 1
        entry["priority_score"] = new

    data["last_updated"] = datetime.now(UTC).isoformat()
    updated_json = json.dumps(data, indent=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(updated_json)
        os.replace(tmp, output_path)
    except Exception:
        os.unlink(tmp)
        raise

    n = len(entries)
    return {
        "n_entries": n,
        "n_updated": n_updated,
        "mean_old_score": sum(old_scores) / max(n, 1),
        "mean_new_score": sum(new_scores) / max(n, 1),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="bulk_priority_update",
        description="Re-score all scan_log entries with a new priority function.",
    )
    parser.add_argument("log", type=Path, metavar="LOG",
                        help="Path to scan_log.json.")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Output path (default: update in-place).")
    args = parser.parse_args(argv)

    summary = update_priorities(args.log, output_path=args.output)
    print(f"Entries:     {summary['n_entries']}")
    print(f"Updated:     {summary['n_updated']}")
    print(f"Mean score:  {summary['mean_old_score']:.3f} → {summary['mean_new_score']:.3f}")
    print(f"Written to:  {summary['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
