"""Compare multiple candidate JSON result files in a unified Markdown report.

Useful for reviewing results from several batch_scan runs or different pipeline
configurations side by side.

Public API
----------
load_and_merge(paths) -> list[dict]
build_comparison_report(rows, *, title, sort_by) -> str
write_comparison_report(rows, output_path, *, title) -> Path
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VALID_SORT_KEYS = {"false_positive_probability", "rank_score", "period_days"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_merge(paths: list[Path]) -> list[dict[str, Any]]:
    """Load multiple JSON result files and merge into a flat list.

    Each row gains a ``_source_file`` key with the path it came from.

    Args:
        paths: Paths to JSON files.  Each file may be a list of dicts or a
            single dict.

    Returns:
        Flat list of candidate dicts.
    """
    merged: list[dict[str, Any]] = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            tagged = dict(row)
            tagged["_source_file"] = str(p)
            merged.append(tagged)
    return merged


def _fpp(row: dict[str, Any]) -> float | None:
    """Extract FPP from either nested scores or top-level key."""
    val = row.get("scores", {}).get("false_positive_probability")
    if val is None:
        val = row.get("best_fpp")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _sort_key(row: dict[str, Any], sort_by: str) -> float:
    """Return a sortable value; missing fields sort to the end.

    rank_score is negated so that higher scores appear first when sorted ascending.
    """
    if sort_by == "false_positive_probability":
        v = _fpp(row)
        return float("inf") if v is None else float(v)
    elif sort_by == "rank_score":
        v = row.get("rank_score")
        return float("-inf") if v is None else -float(v)
    else:  # period_days
        v = row.get("period_days")
        return float("inf") if v is None else float(v)


def build_comparison_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Candidate Comparison",
    sort_by: str = "false_positive_probability",
) -> str:
    """Build a Markdown comparison report for a list of candidates.

    Args:
        rows: Candidate dicts (from :func:`load_and_merge` or any source).
        title: Report heading.
        sort_by: Column to sort by — one of ``"false_positive_probability"``,
            ``"rank_score"``, or ``"period_days"``.

    Returns:
        Markdown string.

    Raises:
        ValueError: If ``sort_by`` is not one of the supported keys.
    """
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(
            f"sort_by must be one of {sorted(_VALID_SORT_KEYS)!r}, got {sort_by!r}"
        )

    if not rows:
        return "_No candidates._\n"

    sorted_rows = sorted(rows, key=lambda r: _sort_key(r, sort_by))

    header = f"# {title}\n"
    table_header = "| Candidate ID | Target | Period (d) | FPP | Pathway |"
    sep           = "| --- | --- | --- | --- | --- |"
    lines = [header, table_header, sep]

    for row in sorted_rows:
        cid     = row.get("candidate_id") or "—"
        target  = row.get("target_id") or "—"
        period  = row.get("period_days")
        fpp_val = _fpp(row)
        pathway = row.get("pathway") or row.get("best_pathway") or "—"

        period_str = f"{period:.4f}" if isinstance(period, float) else "—"
        fpp_str    = f"{fpp_val:.4f}" if isinstance(fpp_val, float) else "—"

        lines.append(f"| {cid} | {target} | {period_str} | {fpp_str} | {pathway} |")

    return "\n".join(lines) + "\n"


def write_comparison_report(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Candidate Comparison",
) -> Path:
    """Write a Markdown comparison report to a file.

    Args:
        rows: Candidate dicts.
        output_path: Destination Markdown file.
        title: Report heading.

    Returns:
        Path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_comparison_report(rows, title=title))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="compare_candidates",
        description="Compare multiple candidate JSON result files in a Markdown report.",
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="JSON result files to merge and compare.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write Markdown report to this file (default: stdout).",
    )
    parser.add_argument(
        "--sort-by",
        default="false_positive_probability",
        choices=sorted(_VALID_SORT_KEYS),
        metavar="KEY",
        help="Column to sort by (default: false_positive_probability).",
    )
    parser.add_argument(
        "--title",
        default="Candidate Comparison",
        metavar="TITLE",
        help="Report title.",
    )
    args = parser.parse_args(argv)

    rows = load_and_merge(args.files)
    report = build_comparison_report(rows, title=args.title, sort_by=args.sort_by)

    if args.output:
        write_comparison_report(rows, args.output, title=args.title)
        print(f"Report written to {args.output}")
    else:
        print(report, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
