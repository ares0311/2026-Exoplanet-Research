"""Generate a full prioritization report across multiple transit candidates.

Combines significance ranking, cross-reference status, and follow-up urgency
into a single comprehensive Markdown report for session planning.

Public API
----------
CandidatePriority(tic_id, period_days, fpp, snr, pathway, rank_score,
                  urgency, action_summary, flag)
build_prioritization_report(rows, *, title, top_n) -> str
write_prioritization_report(rows, output_path, *, title) -> Path
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidatePriority:
    tic_id: int | None
    period_days: float | None
    fpp: float | None
    snr: float | None
    pathway: str
    rank_score: float
    urgency: str  # "URGENT" | "MODERATE" | "LOW"
    action_summary: str
    flag: str  # "OK" | "LOW_QUALITY" | "NEEDS_DATA"


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _compute_rank(row: dict) -> float:
    fpp = _safe_float(
        row.get("false_positive_probability")
        or (row.get("scores") or {}).get("false_positive_probability")
    ) or 0.5
    snr = _safe_float(row.get("snr") or row.get("detection_snr")) or 0.0
    novelty = _safe_float(
        row.get("novelty_score")
        or (row.get("scores") or {}).get("novelty_score")
    ) or 0.5
    dc = _safe_float(
        row.get("detection_confidence")
        or (row.get("scores") or {}).get("detection_confidence")
    ) or 0.5

    snr_norm = min(snr / 20.0, 1.0)
    return 0.40 * (1 - fpp) + 0.25 * snr_norm + 0.20 * novelty + 0.15 * dc


def _urgency(fpp: float | None, pathway: str) -> str:
    if fpp is not None and fpp < 0.10 and "tfop" in pathway:
        return "URGENT"
    if fpp is not None and fpp < 0.30:
        return "MODERATE"
    return "LOW"


def _action_summary(row: dict) -> str:
    pathway = str(row.get("pathway") or "unknown")
    fpp = _safe_float(
        row.get("false_positive_probability")
        or (row.get("scores") or {}).get("false_positive_probability")
    )
    parts: list[str] = []
    if "tfop_ready" in pathway:
        parts.append("Submit to TFOP WG")
    elif "planet_hunters" in pathway:
        parts.append("Post to PH Talk")
    elif "github_only" in pathway:
        parts.append("Archive on GitHub")
    if fpp is not None and fpp > 0.50:
        parts.append("review false-positive evidence")
    if not parts:
        parts.append("Manual review required")
    return "; ".join(parts)


def _prioritize_row(row: dict) -> CandidatePriority:
    tic_id: int | None = None
    with contextlib.suppress(TypeError, ValueError):
        raw = row.get("tic_id")
        if raw is not None:
            tic_id = int(raw)

    period = _safe_float(row.get("period_days"))
    fpp = _safe_float(
        row.get("false_positive_probability")
        or (row.get("scores") or {}).get("false_positive_probability")
    )
    snr = _safe_float(row.get("snr") or row.get("detection_snr"))
    pathway = str(row.get("pathway") or "unknown")
    rank_score = _compute_rank(row)
    urgency = _urgency(fpp, pathway)
    action = _action_summary(row)

    if fpp is None or snr is None:
        flag = "NEEDS_DATA"
    elif fpp > 0.50:
        flag = "LOW_QUALITY"
    else:
        flag = "OK"

    return CandidatePriority(
        tic_id=tic_id,
        period_days=period,
        fpp=fpp,
        snr=snr,
        pathway=pathway,
        rank_score=round(rank_score, 4),
        urgency=urgency,
        action_summary=action,
        flag=flag,
    )


def build_prioritization_report(
    rows: list[dict],
    *,
    title: str = "Candidate Prioritization Report",
    top_n: int | None = None,
) -> str:
    """Build a Markdown prioritization report.

    Args:
        rows: Pipeline output rows.
        title: Report title.
        top_n: Limit to top N candidates; None = all.

    Returns:
        Markdown string.
    """
    if not rows:
        return f"# {title}\n\n_No candidates to prioritize._"

    priorities = sorted(
        [_prioritize_row(r) for r in rows],
        key=lambda p: p.rank_score,
        reverse=True,
    )
    if top_n is not None:
        priorities = priorities[:top_n]

    n_urgent = sum(1 for p in priorities if p.urgency == "URGENT")
    n_moderate = sum(1 for p in priorities if p.urgency == "MODERATE")
    n_low = sum(1 for p in priorities if p.urgency == "LOW")

    lines = [
        f"# {title}\n",
        f"**Candidates**: {len(priorities)} | "
        f"Urgent: {n_urgent} | Moderate: {n_moderate} | Low: {n_low}\n",
        "",
        "## Priority Table\n",
        "| Rank | TIC ID | Period (d) | FPP | SNR | Urgency | Score | Action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, p in enumerate(priorities, 1):
        tic_str = str(p.tic_id) if p.tic_id is not None else "—"
        period_str = f"{p.period_days:.4f}" if p.period_days is not None else "—"
        fpp_str = f"{p.fpp:.3f}" if p.fpp is not None else "—"
        snr_str = f"{p.snr:.1f}" if p.snr is not None else "—"
        lines.append(
            f"| {rank} | {tic_str} | {period_str} | {fpp_str} | {snr_str} | "
            f"`{p.urgency}` | {p.rank_score:.4f} | {p.action_summary} |"
        )

    # Urgent section
    if n_urgent > 0:
        lines += ["", "## Urgent Actions Required\n"]
        for p in priorities:
            if p.urgency != "URGENT":
                continue
            tic_str = str(p.tic_id) if p.tic_id is not None else "Unknown"
            lines.append(f"- **TIC {tic_str}**: {p.action_summary}")

    return "\n".join(lines)


def write_prioritization_report(
    rows: list[dict],
    output_path: str | Path,
    *,
    title: str = "Candidate Prioritization Report",
) -> Path:
    """Write a prioritization report to a Markdown file.

    Args:
        rows: Pipeline output rows.
        output_path: Output file path.
        title: Report title.

    Returns:
        Path to the written file.
    """
    import os
    import tempfile

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = build_prioritization_report(rows, title=title)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, p)
    except Exception:
        with contextlib.suppress(OSError):
            if tmp:
                os.unlink(tmp)
        raise
    return p


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build candidate prioritization report.")
    parser.add_argument("input", help="Candidate JSON file.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default="Candidate Prioritization Report")
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        rows = [rows]

    report = build_prioritization_report(rows, title=args.title, top_n=args.top_n)
    if args.output:
        write_prioritization_report(rows, args.output, title=args.title)
        print(f"Report written to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
