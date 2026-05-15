"""Batch scan multiple TESS/Kepler targets from a TIC ID list.

Reads a plain-text or CSV file of TIC IDs (one per line; lines starting with
``#`` are ignored; CSV files use the first numeric column), runs the full
detection pipeline on each, and writes a JSON results file.

Supports ``--resume``: previously completed TIC IDs are skipped by reading the
existing output file, enabling incremental runs over large target lists.

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
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    tic_ids = read_tic_ids(args.targets)
    if not tic_ids:
        print("No valid TIC IDs found in the input file.", file=sys.stderr)
        return 1

    print(f"Loaded {len(tic_ids)} TIC IDs from {args.targets}", file=sys.stderr)
    batch_scan(
        tic_ids,
        output_path=args.output,
        mission=args.mission,
        min_snr=args.min_snr,
        max_peaks=args.max_peaks,
        scorer=args.scorer,
        model_path=args.model_path,
        resume=args.resume,
    )
    print(f"Results written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
