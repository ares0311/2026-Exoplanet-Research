"""Export ranked candidate results to CSV and Markdown formats.

Reads JSON files produced by ``exo --output`` (or ``rank_candidates.py``) and
writes them to alternative formats suited for sharing or reporting.

Public API
----------
to_csv(rows, path) -> Path
to_markdown_table(rows) -> str
to_summary_stats(rows) -> dict
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Columns included in export outputs (in display order)
_EXPORT_COLUMNS: list[tuple[str, str]] = [
    ("candidate_id", "Candidate ID"),
    ("target_id", "Target"),
    ("period_days", "Period (d)"),
    ("depth_ppm", "Depth (ppm)"),
    ("snr", "SNR"),
    ("false_positive_probability", "FPP"),
    ("detection_confidence", "DC"),
    ("provenance_score", "Provenance"),
    ("rank_score", "Rank Score"),
    ("pathway", "Pathway"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get(row: dict[str, Any], key: str) -> Any:
    """Extract top-level key or scores sub-key."""
    if key in row:
        return row[key]
    return row.get("scores", {}).get(key)


def to_csv(rows: list[dict[str, Any]], path: Path | str) -> Path:
    """Write candidate rows to a CSV file.

    Args:
        rows: Candidate dicts (as from ``rank_candidates.load_candidates``).
        path: Output CSV file path.

    Returns:
        Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [col for col, _ in _EXPORT_COLUMNS]
    display_headers = {col: label for col, label in _EXPORT_COLUMNS}

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(display_headers)  # type: ignore[arg-type]
        for row in rows:
            flat = {col: _get(row, col) for col, _ in _EXPORT_COLUMNS}
            writer.writerow(flat)  # type: ignore[arg-type]

    return path


def to_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Return a GitHub-flavored Markdown table of candidates.

    Args:
        rows: Candidate dicts.

    Returns:
        Markdown string with header + separator + data rows.
    """
    if not rows:
        return "_No candidates._\n"

    labels = [label for _, label in _EXPORT_COLUMNS]
    header = "| " + " | ".join(labels) + " |"
    sep = "| " + " | ".join("---" for _ in labels) + " |"

    lines = [header, sep]
    for row in rows:
        cells: list[str] = []
        for col, _ in _EXPORT_COLUMNS:
            val = _get(row, col)
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if col in ("period_days",) else f"{val:.3f}")
            elif val is None:
                cells.append("—")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def to_summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics over a list of candidate rows.

    Returns:
        Dict with keys: ``n_candidates``, ``mean_fpp``, ``min_fpp``,
        ``max_rank_score``, ``pathway_counts``.
    """
    fpps = [_get(r, "false_positive_probability") for r in rows]
    fpps_valid = [f for f in fpps if isinstance(f, float)]

    ranks = [r.get("rank_score") for r in rows]
    ranks_valid = [r for r in ranks if isinstance(r, float)]

    pathway_counts: dict[str, int] = {}
    for row in rows:
        p = str(row.get("pathway") or row.get("best_pathway") or "unknown")
        pathway_counts[p] = pathway_counts.get(p, 0) + 1

    return {
        "n_candidates": len(rows),
        "mean_fpp": sum(fpps_valid) / len(fpps_valid) if fpps_valid else None,
        "min_fpp": min(fpps_valid) if fpps_valid else None,
        "max_rank_score": max(ranks_valid) if ranks_valid else None,
        "pathway_counts": pathway_counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="export_candidates",
        description="Export ranked exo-toolkit candidates to CSV and/or Markdown.",
    )
    parser.add_argument("file", type=Path, metavar="FILE",
                        help="JSON results file (exo --output or rank_candidates).")
    parser.add_argument("--csv", type=Path, default=None, metavar="OUT",
                        help="Write CSV to this path.")
    parser.add_argument("--markdown", type=Path, default=None, metavar="OUT",
                        help="Write Markdown table to this path.")
    parser.add_argument("--stats", action="store_true",
                        help="Print summary statistics.")
    args = parser.parse_args(argv)

    data = json.loads(args.file.read_text())
    rows = data if isinstance(data, list) else [data]

    if not rows:
        print("No candidates found.", file=sys.stderr)
        return 1

    if args.csv:
        to_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")

    if args.markdown:
        args.markdown.write_text(to_markdown_table(rows))
        print(f"Markdown written to {args.markdown}")

    if args.stats:
        stats = to_summary_stats(rows)
        for k, v in stats.items():
            print(f"{k}: {v}")

    if not any([args.csv, args.markdown, args.stats]):
        print(to_markdown_table(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
