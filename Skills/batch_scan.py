"""Batch scan multiple TESS/Kepler targets from a TIC ID list.

Reads a plain-text or CSV file of TIC IDs (one per line; lines starting with
``#`` are ignored; CSV files use the first numeric column), runs the full
detection pipeline on each, and writes a JSON results file.

Supports ``--resume``: previously completed TIC IDs are skipped by reading the
existing output file, enabling incremental runs over large target lists.

The CLI entry point writes a structured completion record via
``Skills/run_report.py`` after each run (AGENTS.md Run Report Policy, Rule 7)
and commits/pushes only that record.

Public API
----------
read_tic_ids(path) -> list[int]
batch_scan(tic_ids, *, output_path, mission, min_snr, max_peaks, scorer,
           model_path, resume, run_pipeline_fn) -> list[dict]

CLI usage
---------
    python Skills/batch_scan.py targets.txt --output results.json
    python Skills/batch_scan.py targets.txt --output results.json --resume
    python Skills/batch_scan.py targets.csv --output results.json --mission TESS
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_toolkit.hunter_history import build_manual_scan_source
from exo_toolkit.search_lifecycle import HunterStore
from Skills.run_report import RunReport, report_path_for, run_and_commit_report

# ---------------------------------------------------------------------------
# TIC ID reader
# ---------------------------------------------------------------------------


def read_tic_ids(path: Path) -> list[int]:
    """Parse TIC IDs from a plain-text or CSV file.

    Rules:
    - Lines starting with ``#`` are comments and are skipped.
    - Empty lines are skipped.
    - For CSV files (path ends in ``.csv``), the first token on each non-comment
      line that parses as a positive integer is used.
    - For plain-text files, each non-comment line must be a single integer.
    """
    ids: list[int] = []
    is_csv = path.suffix.lower() == ".csv"
    header_skipped = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if is_csv and not header_skipped:
            # Skip header row if first token isn't numeric
            first = line.split(",")[0].strip()
            if not first.lstrip("-").isdigit():
                header_skipped = True
                continue
        tokens = line.split(",") if is_csv else [line]
        for tok in tokens:
            tok = tok.strip()
            if tok.lstrip("-").isdigit():
                val = int(tok)
                if val > 0:
                    ids.append(val)
                break

    return ids


# ---------------------------------------------------------------------------
# Batch scan
# ---------------------------------------------------------------------------


def batch_scan(
    tic_ids: list[int],
    *,
    output_path: Path,
    mission: str = "TESS",
    min_snr: float = 5.0,
    max_peaks: int = 5,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    resume: bool = False,
    run_pipeline_fn: Callable[..., list[dict[str, Any]]] | None = None,
    new_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the detection pipeline on a list of TIC IDs and write results.

    Args:
        tic_ids: List of TESS Input Catalog IDs to scan.
        output_path: Path to write JSON results.  Created if absent.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        min_snr: Minimum BLS SNR threshold.
        max_peaks: Maximum signals to search per target.
        scorer: ``"bayesian"``, ``"xgboost"``, or ``"ensemble"``.
        model_path: XGBoost model path (required for xgboost/ensemble).
        resume: If True, skip TIC IDs already present in *output_path*.
        run_pipeline_fn: Override the pipeline function (for tests).
        new_entries: Optional mutable list populated in-place with exactly
            the entries scanned in *this* call (excluding anything skipped
            via ``resume``), so callers can distinguish freshly-scanned
            targets from the full accumulated results without changing the
            return contract.

    Returns:
        List of all result dicts written to *output_path*.
    """
    if run_pipeline_fn is None:
        from exo_toolkit.cli import run_pipeline  # noqa: PLC0415
        run_pipeline_fn = run_pipeline

    # Load existing results if resuming
    all_results: list[dict[str, Any]] = []
    completed_tic_ids: set[int] = set()
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        if isinstance(existing, list):
            all_results = existing
            for entry in all_results:
                tid = entry.get("tic_id")
                if tid is not None:
                    completed_tic_ids.add(int(tid))

    remaining = [t for t in tic_ids if t not in completed_tic_ids]
    print(
        f"Scanning {len(remaining)} targets "
        f"({len(tic_ids) - len(remaining)} skipped via --resume).",
        file=sys.stderr,
    )

    for tic_id in remaining:
        target_id = f"TIC {tic_id}"
        entry: dict[str, Any] = {
            "tic_id": tic_id,
            "target_id": target_id,
            "mission": mission,
            "status": "pending",
            "signals": [],
        }
        try:
            signals = run_pipeline_fn(
                target_id,
                mission,  # type: ignore[arg-type]
                min_snr=min_snr,
                max_peaks=max_peaks,
                scorer=scorer,
                model_path=model_path,
            )
            entry["status"] = "candidate_found" if signals else "scanned_clear"
            entry["signals"] = signals
            print(
                f"  {target_id}: {entry['status']} ({len(signals)} signal(s))",
                file=sys.stderr,
            )
        except Exception:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = traceback.format_exc()
            print(f"  {target_id}: ERROR", file=sys.stderr)

        all_results.append(entry)
        if new_entries is not None:
            new_entries.append(entry)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))

    return all_results


# ---------------------------------------------------------------------------
# Hunter durable-history bridge
# ---------------------------------------------------------------------------

_HISTORY_STATUS_MAP = {
    "candidate_found": "candidate_found",
    "scanned_clear": "no_signal",
    "error": "failed",
}


def _manual_scan_history_entry(entry: dict[str, Any], *, mission: str) -> dict[str, Any]:
    """Map one batch_scan result entry to a history-manifest entry."""
    status = _HISTORY_STATUS_MAP.get(str(entry.get("status")), "failed")
    history_entry: dict[str, Any] = {
        "target_id": str(entry.get("target_id")),
        "status": status,
        "searched_at": datetime.now(UTC).isoformat(),
        "mission": mission,
        "ranking_score": 0.0,
        "metrics": {"n_signals": len(entry.get("signals") or [])},
        "result": entry,
    }
    if status == "failed":
        history_entry["error_message"] = str(entry.get("error") or "unknown error")
    return history_entry


def _bridge_manual_scan_to_hunter(
    *,
    log_path: Path,
    mission: str,
    entries: list[dict[str, Any]],
    started_at: datetime,
    completed_at: datetime,
    hunter_db_path: Path,
) -> dict[str, int] | None:
    """Durably record a completed batch_scan CLI run in Hunter's history.

    Standalone batch_scan.py CLI runs write only to a local JSON results
    file by default, invisible to EXO-Hunter's durable
    target_search_history -- a real bypass of "no production command
    bypasses the canonical optimizer or durable pipeline." This registers
    exactly the targets scanned in *this* run using the same fail-closed
    import_history_manifest() path the seven curated legacy-log imports
    use. Failure here is reported loudly but does not fail an otherwise-
    successful scan run, matching this file's existing run-report
    commit/push-failure precedent.
    """
    if not entries:
        return None
    try:
        source_root = log_path.resolve().parent
        source = build_manual_scan_source(
            script="batch_scan",
            log_path=log_path,
            entries=[
                _manual_scan_history_entry(entry, mission=mission) for entry in entries
            ],
            started_at=started_at,
            completed_at=completed_at,
            method_or_data=f"{mission} manual batch scan",
            source_root=source_root,
        )
        store = HunterStore(hunter_db_path)
        summary = store.import_history_manifest(
            {"schema_version": 1, "sources": [source]},
            source_root=source_root,
        )
        print(
            f"Hunter durable history: {summary['sources_created']} source(s), "
            f"{summary['events_created']} event(s) recorded to {hunter_db_path}",
            file=sys.stderr,
            flush=True,
        )
        return summary
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: manual scan was not bridged to Hunter durable history: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Run Report
# ---------------------------------------------------------------------------


def _write_run_report(
    *,
    started_at: str,
    elapsed_seconds: float,
    results: list[dict[str, Any]],
    output_path: Path,
    git_run_fn: Any = None,
) -> None:
    """Append and publish one batch_scan completion report (AGENTS.md Rule 7)."""
    items_failed = sum(1 for entry in results if entry.get("status") == "error")
    report = RunReport(
        script="batch_scan",
        status="success" if items_failed == 0 else "partial",
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed_seconds,
        items_processed=len(results),
        items_written=len(results) - items_failed,
        items_failed=items_failed,
        output_paths=(str(output_path),),
    )
    path = report_path_for("batch_scan")
    kwargs: dict[str, Any] = {}
    if git_run_fn is not None:
        kwargs["run_fn"] = git_run_fn
    ok = run_and_commit_report(report, path, **kwargs)
    if ok:
        print(f"Run report committed and pushed: {path}", file=sys.stderr, flush=True)
    else:
        print(
            f"Warning: run report written to {path} but commit/push failed",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None, *, git_run_fn: Any = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="batch_scan",
        description="Run the exo pipeline over a list of TIC IDs.",
    )
    parser.add_argument("targets", type=Path, help="Text or CSV file of TIC IDs.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON file path."
    )
    parser.add_argument(
        "--mission", default="TESS", choices=["TESS", "Kepler", "K2"],
    )
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument(
        "--scorer", default="bayesian", choices=["bayesian", "xgboost", "ensemble"]
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip TIC IDs already in the output file.",
    )
    parser.add_argument(
        "--hunter-db",
        type=Path,
        default=Path("data/hunter_searches.sqlite3"),
        help="EXO-Hunter durable SQLite database to record this manual scan in",
    )
    parser.add_argument(
        "--no-hunter-bridge",
        action="store_true",
        help="Skip recording this manual scan in Hunter's durable target_search_history",
    )
    args = parser.parse_args(argv)

    tic_ids = read_tic_ids(args.targets)
    if not tic_ids:
        print("No valid TIC IDs found in the input file.", file=sys.stderr)
        return 1

    print(f"Loaded {len(tic_ids)} TIC IDs from {args.targets}", file=sys.stderr)
    started_at_dt = datetime.now(UTC)
    started_at = started_at_dt.isoformat()
    start = time.monotonic()
    new_entries: list[dict[str, Any]] = []
    results = batch_scan(
        tic_ids,
        output_path=args.output,
        mission=args.mission,
        min_snr=args.min_snr,
        max_peaks=args.max_peaks,
        scorer=args.scorer,
        model_path=args.model_path,
        resume=args.resume,
        new_entries=new_entries,
    )
    elapsed = time.monotonic() - start
    print(f"Results written to {args.output}", file=sys.stderr)
    if not args.no_hunter_bridge:
        _bridge_manual_scan_to_hunter(
            log_path=args.output,
            mission=args.mission,
            entries=new_entries,
            started_at=started_at_dt,
            completed_at=datetime.now(UTC),
            hunter_db_path=args.hunter_db,
        )
    _write_run_report(
        started_at=started_at,
        elapsed_seconds=elapsed,
        results=results,
        output_path=args.output,
        git_run_fn=git_run_fn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
