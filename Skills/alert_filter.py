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
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else [data]
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

    filtered = apply_filters(
        args.file,
        output_path=args.output,
        fpp_max=args.fpp_max,
        pathway=args.pathway,
        min_signals=args.min_signals,
        min_rank_score=args.min_rank,
        min_snr=args.min_snr,
    )

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
