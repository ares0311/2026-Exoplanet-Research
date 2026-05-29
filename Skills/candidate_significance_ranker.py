"""Rank candidates by composite significance combining SNR, FPP, and novelty.

Public API
----------
SignificanceResult(tic_id, period_days, snr, fpp, novelty_score,
                   significance_score, rank, flag)
rank_by_significance(rows, *, top_n, min_snr, max_fpp) -> list[SignificanceResult]
format_significance_table(results) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SignificanceResult:
    tic_id: int | None
    period_days: float | None
    snr: float | None
    fpp: float | None
    novelty_score: float | None
    significance_score: float
    rank: int
    flag: str  # "OK" | "LOW_SNR" | "HIGH_FPP" | "FILTERED"


def _safe_float(val, default: float | None = None) -> float | None:
    if val is None:
        return default
    with contextlib.suppress(TypeError, ValueError):
        return float(val)
    return default


def _significance_score(snr: float | None, fpp: float | None,
                         novelty: float | None) -> float:
    """Composite score: higher is more significant."""
    snr_norm = min((snr or 0.0) / 20.0, 1.0)        # saturates at SNR=20
    fpp_contrib = 1.0 - (fpp if fpp is not None else 0.5)
    nov_contrib = (novelty if novelty is not None else 0.5)
    return 0.50 * snr_norm + 0.35 * fpp_contrib + 0.15 * nov_contrib


def rank_by_significance(
    rows: list[dict],
    *,
    top_n: int | None = None,
    min_snr: float | None = None,
    max_fpp: float | None = None,
) -> list[SignificanceResult]:
    """Rank candidate rows by composite significance score.

    Args:
        rows: Candidate dicts with optional keys snr, false_positive_probability,
              novelty_score, tic_id, period_days.
        top_n: Return only the top N results; None returns all.
        min_snr: Exclude rows with SNR below this threshold.
        max_fpp: Exclude rows with FPP above this threshold.

    Returns:
        List of SignificanceResult sorted descending by significance_score.
    """
    results: list[tuple[float, dict]] = []
    for row in rows:
        snr = _safe_float(row.get("snr") or row.get("detection_snr"))
        fpp = _safe_float(
            row.get("false_positive_probability")
            or (row.get("scores") or {}).get("false_positive_probability")
        )
        novelty = _safe_float(row.get("novelty_score")
                               or (row.get("scores") or {}).get("novelty_score"))
        score = _significance_score(snr, fpp, novelty)
        results.append((score, row))

    results.sort(key=lambda x: x[0], reverse=True)

    out: list[SignificanceResult] = []
    rank = 0
    for score, row in results:
        snr = _safe_float(row.get("snr") or row.get("detection_snr"))
        fpp = _safe_float(
            row.get("false_positive_probability")
            or (row.get("scores") or {}).get("false_positive_probability")
        )
        novelty = _safe_float(row.get("novelty_score")
                               or (row.get("scores") or {}).get("novelty_score"))
        tic_id_raw = row.get("tic_id")
        tic_id: int | None = None
        with contextlib.suppress(TypeError, ValueError):
            tic_id = int(tic_id_raw) if tic_id_raw is not None else None

        filtered = False
        if min_snr is not None and (snr is None or snr < min_snr):
            filtered = True
        if max_fpp is not None and (fpp is None or fpp > max_fpp):
            filtered = True

        if filtered:
            flag = "FILTERED"
        elif snr is not None and snr < 7.1:
            flag = "LOW_SNR"
        elif fpp is not None and fpp > 0.50:
            flag = "HIGH_FPP"
        else:
            flag = "OK"

        if not filtered:
            rank += 1

        out.append(SignificanceResult(
            tic_id=tic_id,
            period_days=_safe_float(row.get("period_days")),
            snr=snr,
            fpp=fpp,
            novelty_score=novelty,
            significance_score=round(score, 4),
            rank=rank if not filtered else 0,
            flag=flag,
        ))

        if top_n is not None and rank >= top_n:
            break

    return out


def format_significance_table(results: list[SignificanceResult]) -> str:
    """Format ranked results as a Markdown table.

    Args:
        results: List of SignificanceResult from rank_by_significance.

    Returns:
        Markdown string.
    """
    if not results:
        return "## Candidate Significance Ranking\n\n_No results._"

    lines = [
        "## Candidate Significance Ranking\n",
        f"**{len(results)} candidate(s) ranked**\n",
        "",
        "| Rank | TIC ID | Period (d) | SNR | FPP | Novelty | Score | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rank_str = str(r.rank) if r.rank > 0 else "—"
        tic_str = str(r.tic_id) if r.tic_id is not None else "—"
        period_str = f"{r.period_days:.4f}" if r.period_days is not None else "—"
        snr_str = f"{r.snr:.1f}" if r.snr is not None else "—"
        fpp_str = f"{r.fpp:.3f}" if r.fpp is not None else "—"
        nov_str = f"{r.novelty_score:.3f}" if r.novelty_score is not None else "—"
        lines.append(
            f"| {rank_str} | {tic_str} | {period_str} | {snr_str} | "
            f"{fpp_str} | {nov_str} | {r.significance_score:.4f} | `{r.flag}` |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Rank candidates by significance.")
    parser.add_argument("input", help="JSON file of candidate rows.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-snr", type=float, default=None)
    parser.add_argument("--max-fpp", type=float, default=None)
    args = parser.parse_args(argv)

    from pathlib import Path
    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        rows = [rows]
    results = rank_by_significance(rows, top_n=args.top_n,
                                   min_snr=args.min_snr, max_fpp=args.max_fpp)
    print(format_significance_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
