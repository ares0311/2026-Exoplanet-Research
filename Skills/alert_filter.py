"""Filter batch_scan or star_scanner JSON results by threshold criteria.

Useful for quickly triaging large result sets — e.g. "show me only candidates
with FPP < 0.2 and at least two signals that reached the tfop_ready pathway."

Public API
----------
filter_candidates(rows, *, fpp_max, pathway, min_signals, min_rank_score,
                  min_snr) -> list[dict]
apply_filters(path, *, output_path, ...) -> list[dict]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SENTINEL = object()  # marks "not supplied" for optional thresholds


def _operator_file_error(path: Path, error: Exception) -> str:
    """Return a traceback-free CLI message for missing or unreadable inputs."""
    if isinstance(error, FileNotFoundError):
        detail = "does not exist"
    elif isinstance(error, json.JSONDecodeError):
        detail = "is not valid complete JSON"
    else:
        detail = f"could not be read ({error})"
    return (
        f"Input file {detail}: {path}\n"
        "If this is a live discovery scan log, let Skills/star_scanner.py finish "
        "successfully before running alert_filter.py. If the scan was stopped "
        "or suspended, rerun the scanner from main and then filter the completed log."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_candidates(
    rows: list[dict[str, Any]],
    *,
    fpp_max: float | None = None,
    pathway: str | None = None,
    min_signals: int | None = None,
    min_rank_score: float | None = None,
    min_snr: float | None = None,
) -> list[dict[str, Any]]:
    """Return rows that satisfy all supplied threshold criteria.

    Criteria are combined with AND logic — a row must pass every supplied
    criterion to be included.  Omitted criteria (``None``) are not checked.

    Args:
        rows: List of candidate dicts from ``exo --output``, ``batch_scan``,
            or ``star_scanner`` output.
        fpp_max: Maximum false-positive probability (inclusive).
        pathway: Required submission pathway string (exact match).
        min_signals: Minimum number of detected signals (``n_signals`` key).
        min_rank_score: Minimum composite rank score.
        min_snr: Minimum SNR of the best signal.

    Returns:
        Filtered list of rows.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        fpp = _fpp(row)
        if fpp_max is not None and (fpp is None or fpp > fpp_max):
            continue
        if pathway is not None:
            row_pathway = str(row.get("pathway") or row.get("best_pathway") or "")
            if row_pathway != pathway:
                continue
        if min_signals is not None:
            n = row.get("n_signals")
            if n is None or int(n) < min_signals:
                continue
        if min_rank_score is not None:
            rs = row.get("rank_score")
            if rs is None or float(rs) < min_rank_score:
                continue
        if min_snr is not None:
            snr = row.get("snr") or row.get("best_snr")
            if snr is None or float(snr) < min_snr:
                continue
        out.append(row)
    return out


def apply_filters(
    path: Path | str,
    *,
    output_path: Path | str | None = None,
    fpp_max: float | None = None,
    pathway: str | None = None,
    min_signals: int | None = None,
    min_rank_score: float | None = None,
    min_snr: float | None = None,
) -> list[dict[str, Any]]:
    """Load a JSON results file, filter it, and optionally write the output.

    Args:
        path: Input JSON file (list or single dict).
        output_path: If supplied, write filtered rows as JSON to this path.
        **criteria: Passed through to :func:`filter_candidates`.

    Returns:
        Filtered list of rows.
    """
    rows = _load_rows(path)
    filtered = filter_candidates(
        rows,
        fpp_max=fpp_max,
        pathway=pathway,
        min_signals=min_signals,
        min_rank_score=min_rank_score,
        min_snr=min_snr,
    )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(filtered, indent=2))
    return filtered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load candidate rows from list, single-result, or star_scanner log JSON."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        return _scan_log_to_rows(data)
    if isinstance(data, dict):
        return [data]
    return []


def _scan_log_to_rows(log: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert ``Skills/star_scanner.py --log`` output into filterable rows."""
    rows: list[dict[str, Any]] = []
    entries = log.get("entries", {})
    if not isinstance(entries, dict):
        return rows

    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "candidate_found":
            continue

        tic_id = entry.get("tic_id")
        target_id = f"TIC {tic_id}" if tic_id is not None else str(key)
        best_fpp = entry.get("best_fpp")
        detection_confidence = entry.get("best_detection_confidence")
        novelty_score = entry.get("best_novelty_score")
        rows.append(
            {
                "candidate_id": target_id,
                "target_id": target_id,
                "tic_id": tic_id,
                "status": entry.get("status"),
                "n_signals": entry.get("n_signals", 0),
                "period_days": entry.get("best_period_days"),
                "best_period_days": entry.get("best_period_days"),
                "pathway": entry.get("best_pathway"),
                "best_pathway": entry.get("best_pathway"),
                "false_positive_probability": best_fpp,
                "best_fpp": best_fpp,
                "snr": entry.get("best_snr"),
                "best_snr": entry.get("best_snr"),
                "depth_ppm": entry.get("best_depth_ppm"),
                "duration_hours": entry.get("best_duration_hours"),
                "transit_count": entry.get("best_transit_count"),
                "provenance_score": entry.get("provenance_score"),
                "priority_score": entry.get("priority_score"),
                "scanned_at": entry.get("scanned_at"),
                "signals": entry.get("signals", []),
                "scores": {
                    "false_positive_probability": best_fpp,
                    "detection_confidence": detection_confidence,
                    "novelty_score": novelty_score,
                },
            }
        )
    return rows


def _fpp(row: dict[str, Any]) -> float | None:
    """Extract FPP from either top-level or scores sub-dict."""
    val = row.get("false_positive_probability") or row.get("best_fpp")
    if val is None:
        val = row.get("scores", {}).get("false_positive_probability")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="alert_filter",
        description="Filter batch_scan/star_scanner JSON results by thresholds.",
    )
    parser.add_argument("file", type=Path, metavar="FILE",
                        help="Input JSON results file.")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Write filtered JSON to this file.")
    parser.add_argument("--fpp-max", type=float, default=None, metavar="X",
                        help="Maximum FPP (e.g. 0.30).")
    parser.add_argument("--pathway", default=None, metavar="P",
                        help="Required pathway (e.g. tfop_ready).")
    parser.add_argument("--min-signals", type=int, default=None, metavar="N",
                        help="Minimum number of signals.")
    parser.add_argument("--min-rank", type=float, default=None, metavar="X",
                        help="Minimum rank score.")
    parser.add_argument("--min-snr", type=float, default=None, metavar="X",
                        help="Minimum SNR.")
    args = parser.parse_args(argv)

    try:
        filtered = apply_filters(
            args.file,
            output_path=args.output,
            fpp_max=args.fpp_max,
            pathway=args.pathway,
            min_signals=args.min_signals,
            min_rank_score=args.min_rank,
            min_snr=args.min_snr,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        filename = getattr(exc, "filename", None)
        path = Path(filename) if filename else args.file
        print(_operator_file_error(path, exc), file=sys.stderr)
        return 2

    if args.output:
        print(f"Filtered {len(filtered)} candidates → {args.output}")
    else:
        print(json.dumps(filtered, indent=2))

    if not filtered:
        print("No candidates matched the filters.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
