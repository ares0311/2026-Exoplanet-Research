"""Generate a Markdown summary report from batch_scan JSON output.

Reads one or more batch_scan output JSON files (lists of dicts with
``tic_id``, ``status``, ``n_signals``, ``best_period_days``, ``best_fpp``,
``best_pathway``) and generates a structured Markdown report.

Public API
----------
load_results(paths) -> list[dict]
build_report(rows, *, title) -> str
write_report(rows, output_path, *, title) -> Path
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Status groupings
# ---------------------------------------------------------------------------

_STATUS_LABEL: dict[str, str] = {
    "candidate_found": "Candidates found",
    "scanned_clear": "Scanned — no signal",
    "no_data": "No data available",
    "error": "Errors",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_results(paths: list[Path | str]) -> list[dict[str, Any]]:
    """Load batch scan result rows from one or more JSON files.

    Each file may contain a single dict or a list of dicts.  Returns a
    flat list with a ``_source_file`` key added for traceability.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            data = [data]
        for row in data:
            row = dict(row)
            row.setdefault("_source_file", str(path))
            rows.append(row)
    return rows


def build_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Batch Scan Summary",
) -> str:
    """Build a Markdown report string from batch scan result rows."""
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Partition rows by status
    by_status: dict[str, list[dict[str, Any]]] = {s: [] for s in _STATUS_LABEL}
    for row in rows:
        status = str(row.get("status", "error"))
        by_status.setdefault(status, []).append(row)

    candidates = by_status.get("candidate_found", [])
    n_total = len(rows)
    n_candidates = len(candidates)
    n_clear = len(by_status.get("scanned_clear", []))
    n_no_data = len(by_status.get("no_data", []))
    n_errors = len(by_status.get("error", []))

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append(f"\n_Generated: {now}_\n")

    # Summary table
    lines.append("## Overview\n")
    lines.append("| Status | Count |")
    lines.append("|--------|------:|")
    lines.append(f"| Total scanned | {n_total} |")
    lines.append(f"| Candidates found | {n_candidates} |")
    lines.append(f"| Scanned clear | {n_clear} |")
    lines.append(f"| No data | {n_no_data} |")
    lines.append(f"| Errors | {n_errors} |")

    # Candidates section
    lines.append("\n## Candidates\n")
    if not candidates:
        lines.append("_No candidates found._\n")
    else:
        lines.append("| TIC ID | Period (d) | FPP | Pathway | Signals |")
        lines.append("|--------|------------|-----|---------|--------:|")
        for row in sorted(candidates, key=lambda r: float(r.get("best_fpp", 1.0))):
            tic = row.get("tic_id", "?")
            period = row.get("best_period_days", float("nan"))
            fpp = row.get("best_fpp", float("nan"))
            pathway = row.get("best_pathway", "?")
            n_sig = row.get("n_signals", 0)
            period_str = f"{period:.4f}" if isinstance(period, float) else str(period)
            fpp_str = f"{fpp:.3f}" if isinstance(fpp, float) else str(fpp)
            lines.append(f"| TIC {tic} | {period_str} | {fpp_str} | {pathway} | {n_sig} |")

    # Errors section (if any)
    if n_errors:
        lines.append("\n## Errors\n")
        for row in by_status.get("error", []):
            tic = row.get("tic_id", "?")
            msg = row.get("error_message", "unknown error")
            lines.append(f"- **TIC {tic}**: {msg}")

    lines.append("")
    return "\n".join(lines)


def write_report(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Batch Scan Summary",
) -> Path:
    """Write Markdown report to a file and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(rows, title=title))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="summary_report",
        description="Generate a Markdown summary report from batch_scan JSON output.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Batch scan JSON output file(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output Markdown file (default: print to stdout).",
    )
    parser.add_argument(
        "--title",
        default="Batch Scan Summary",
        help="Report title.",
    )
    args = parser.parse_args(argv)

    rows = load_results(args.files)
    if not rows:
        print("No results found.", file=sys.stderr)
        return 1

    report = build_report(rows, title=args.title)

    if args.output:
        write_report(rows, args.output, title=args.title)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
