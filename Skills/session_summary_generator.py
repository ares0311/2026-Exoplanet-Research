"""Generate a Markdown summary of a pipeline session's outputs.

Reads a batch_scan or run-summary JSON file and produces a concise
session summary with statistics, top candidates, and next steps.

Public API
----------
SessionSummary(session_id, n_scanned, n_candidates, n_errors,
               top_candidates, elapsed_s, next_steps, flag)
build_session_summary(rows, *, session_id, elapsed_s,
                      top_n) -> SessionSummary
format_session_summary(summary) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    n_scanned: int
    n_candidates: int
    n_errors: int
    n_no_data: int
    top_candidates: tuple[dict, ...]
    elapsed_s: float | None
    next_steps: tuple[str, ...]
    flag: str  # "OK" | "EMPTY" | "HIGH_ERROR_RATE"


def _fpp(row: dict) -> float:
    with contextlib.suppress(TypeError, ValueError):
        v = (row.get("false_positive_probability")
             or (row.get("scores") or {}).get("false_positive_probability"))
        if v is not None:
            return float(v)
    return 1.0


def build_session_summary(
    rows: list[dict],
    *,
    session_id: str = "session",
    elapsed_s: float | None = None,
    top_n: int = 5,
) -> SessionSummary:
    """Build a session summary from pipeline output rows.

    Args:
        rows: List of pipeline output or batch_scan dicts.
        session_id: Label for the session.
        elapsed_s: Total elapsed time in seconds.
        top_n: Number of top candidates to include.

    Returns:
        SessionSummary with statistics and top candidates.
    """
    if not rows:
        return SessionSummary(
            session_id=session_id,
            n_scanned=0,
            n_candidates=0,
            n_errors=0,
            n_no_data=0,
            top_candidates=(),
            elapsed_s=elapsed_s,
            next_steps=("No targets were scanned in this session.",),
            flag="EMPTY",
        )

    n_scanned = len(rows)
    candidates = [r for r in rows if r.get("status") == "candidate_found"
                  or _fpp(r) < 0.50]
    n_candidates = len(candidates)
    n_errors = sum(1 for r in rows if r.get("status") == "error")
    n_no_data = sum(1 for r in rows if r.get("status") == "no_data")

    top = sorted(candidates, key=_fpp)[:top_n]

    next_steps_list: list[str] = []
    if n_candidates > 0:
        next_steps_list.append(
            f"Follow up on {n_candidates} candidate signal(s) — "
            "run vetting and checklist generation."
        )
    if n_errors > n_scanned * 0.2:
        next_steps_list.append("High error rate — check data availability and pipeline config.")
    if n_candidates == 0:
        next_steps_list.append("No candidates found — expand search to more targets or sectors.")
    next_steps_list.append("Archive results with batch_result_archiver.")

    error_rate = n_errors / n_scanned
    flag = "HIGH_ERROR_RATE" if error_rate > 0.2 else "OK"

    return SessionSummary(
        session_id=session_id,
        n_scanned=n_scanned,
        n_candidates=n_candidates,
        n_errors=n_errors,
        n_no_data=n_no_data,
        top_candidates=tuple(top),
        elapsed_s=elapsed_s,
        next_steps=tuple(next_steps_list),
        flag=flag,
    )


def format_session_summary(summary: SessionSummary) -> str:
    """Format a session summary as Markdown.

    Args:
        summary: SessionSummary to format.

    Returns:
        Markdown string.
    """
    elapsed_str = f"{summary.elapsed_s:.1f} s" if summary.elapsed_s is not None else "—"
    lines = [
        f"# Session Summary — {summary.session_id}\n",
        f"**Status**: `{summary.flag}` | "
        f"Scanned: {summary.n_scanned} | "
        f"Candidates: {summary.n_candidates} | "
        f"Errors: {summary.n_errors} | "
        f"No data: {summary.n_no_data} | "
        f"Elapsed: {elapsed_str}\n",
    ]

    if summary.top_candidates:
        lines += [
            "",
            f"## Top {len(summary.top_candidates)} Candidate(s)\n",
            "| TIC ID | Period (d) | FPP | Pathway |",
            "|---|---|---|---|",
        ]
        for row in summary.top_candidates:
            tic = row.get("tic_id", "—")
            period = row.get("period_days")
            period_str = f"{period:.4f}" if period is not None else "—"
            fpp_val = _fpp(row)
            fpp_str = f"{fpp_val:.3f}"
            pathway = row.get("pathway", "—")
            lines.append(f"| {tic} | {period_str} | {fpp_str} | {pathway} |")

    lines += ["", "## Next Steps\n"]
    for step in summary.next_steps:
        lines.append(f"- {step}")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate session summary.")
    parser.add_argument("input", help="Batch scan JSON file.")
    parser.add_argument("--session-id", default="session")
    parser.add_argument("--elapsed", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    from pathlib import Path
    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        rows = [rows]
    summary = build_session_summary(rows, session_id=args.session_id,
                                    elapsed_s=args.elapsed, top_n=args.top_n)
    print(format_session_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
