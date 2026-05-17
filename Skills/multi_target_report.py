"""Generate a combined Markdown report for multiple pipeline candidates.

Combines a results list with optional CandidateTimeline data into a
structured multi-section report.

Public API
----------
build_multi_target_report(rows, *, timeline_path, title) -> str
write_multi_target_report(rows, output_path, *, timeline_path, title) -> Path
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fpp(row: dict[str, Any]) -> float | None:
    v = row.get("scores", {}).get("false_positive_probability")
    if v is None:
        v = row.get("best_fpp")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _summary_table(rows: list[dict[str, Any]]) -> str:
    header = "| Candidate | Target | Period (d) | FPP | Pathway | Rank |"
    sep    = "| --- | --- | --- | --- | --- | --- |"
    lines  = [header, sep]
    for row in sorted(rows, key=lambda r: _fpp(r) or 1.0):
        cid     = row.get("candidate_id", "—")
        target  = row.get("target_id", "—")
        period  = row.get("period_days")
        fpp_val = _fpp(row)
        pathway = row.get("pathway") or row.get("best_pathway") or "—"
        rank    = row.get("rank_score")
        pstr    = f"{period:.4f}" if isinstance(period, float) else "—"
        fstr    = f"{fpp_val:.4f}" if isinstance(fpp_val, float) else "—"
        rstr    = f"{rank:.3f}"   if isinstance(rank, float)   else "—"
        lines.append(f"| {cid} | {target} | {pstr} | {fstr} | {pathway} | {rstr} |")
    return "\n".join(lines)


def _timeline_section(
    candidate_id: str,
    timeline_data: dict[str, Any],
) -> str:
    entries = timeline_data.get("entries", {}).get(candidate_id, [])
    if not entries:
        return f"_No timeline entries for {candidate_id}._"

    header = "| Run | FPP | Pathway | Scorer |"
    sep    = "| --- | --- | --- | --- |"
    lines  = [header, sep]
    for e in entries[-5:]:  # last 5 runs
        run_at  = e.get("run_at", "—")[:10]
        fpp_val = e.get("fpp")
        pathway = e.get("pathway", "—")
        scorer  = e.get("scorer", "—")
        fstr    = f"{fpp_val:.4f}" if isinstance(fpp_val, float) else "—"
        lines.append(f"| {run_at} | {fstr} | {pathway} | {scorer} |")
    return "\n".join(lines)


def build_multi_target_report(
    rows: list[dict[str, Any]],
    *,
    timeline_path: Path | str | None = None,
    title: str = "Multi-Target Report",
) -> str:
    """Build a Markdown report covering multiple pipeline candidates.

    Args:
        rows: Candidate dicts from one or more pipeline output files.
        timeline_path: Path to a ``CandidateTimeline`` JSON file.  If
            provided, each candidate section includes its score history.
        title: Report heading.

    Returns:
        Markdown string.
    """
    if not rows:
        return "_No candidates._\n"

    timeline_data: dict[str, Any] = {}
    if timeline_path is not None:
        p = Path(timeline_path)
        if p.exists():
            timeline_data = json.loads(p.read_text())

    n_candidates = len(rows)
    n_paths = {}
    for row in rows:
        pw = row.get("pathway") or row.get("best_pathway") or "unknown"
        n_paths[pw] = n_paths.get(pw, 0) + 1

    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- Total candidates: {n_candidates}",
    ]
    for pw, cnt in sorted(n_paths.items()):
        lines.append(f"  - {pw}: {cnt}")

    lines += ["", "## Summary Table", "", _summary_table(rows)]

    for row in sorted(rows, key=lambda r: _fpp(r) or 1.0):
        cid = row.get("candidate_id", "unknown")
        target = row.get("target_id", "—")
        fpp_val = _fpp(row)
        pathway = row.get("pathway") or row.get("best_pathway") or "—"
        period = row.get("period_days")

        lines += [
            "",
            "---",
            "",
            f"## {cid}",
            "",
            f"**Target:** {target}  "
            f"**Period:** {period:.4f} d  "
            f"**FPP:** {fpp_val:.4f}  "
            f"**Pathway:** {pathway}"
            if isinstance(period, float) and isinstance(fpp_val, float)
            else f"**Target:** {target}  **Pathway:** {pathway}",
        ]

        if timeline_data:
            lines += ["", "### Score History", "", _timeline_section(cid, timeline_data)]

    return "\n".join(lines) + "\n"


def write_multi_target_report(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    timeline_path: Path | str | None = None,
    title: str = "Multi-Target Report",
) -> Path:
    """Write a multi-target Markdown report to a file.

    Args:
        rows: Candidate dicts.
        output_path: Destination Markdown file.
        timeline_path: Optional path to a ``CandidateTimeline`` JSON file.
        title: Report heading.

    Returns:
        Path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_multi_target_report(
        rows, timeline_path=timeline_path, title=title
    ))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_target_report",
        description="Combined Markdown report for multiple pipeline candidates.",
    )
    parser.add_argument("files", type=Path, nargs="+", metavar="FILE",
                        help="Pipeline JSON output files.")
    parser.add_argument("--timeline", type=Path, default=None, metavar="FILE",
                        help="CandidateTimeline JSON file.")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Output Markdown file (default: stdout).")
    parser.add_argument("--title", default="Multi-Target Report", metavar="TITLE")
    args = parser.parse_args(argv)

    rows = []
    for f in args.files:
        data = json.loads(f.read_text())
        rows.extend(data if isinstance(data, list) else [data])

    report = build_multi_target_report(
        rows, timeline_path=args.timeline, title=args.title
    )

    if args.output:
        write_multi_target_report(
            rows, args.output, timeline_path=args.timeline, title=args.title
        )
        print(f"Report written to {args.output}")
    else:
        print(report, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
