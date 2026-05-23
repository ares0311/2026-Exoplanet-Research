"""Generate a compact Markdown summary card from a scored candidate dict.

Formats the key fields from an ``exo --output`` JSON row into a single
compact block suitable for quick terminal review.  Intentionally narrow —
for full multi-section reports use ``candidate_report_card.py``.

Public API
----------
SummaryCardResult(tic_id, period_days, depth_ppm, fpp, pathway,
                  top_flags, formatted_card, flag)
build_summary_card(candidate_dict) -> SummaryCardResult
format_summary_card(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryCardResult:
    tic_id: str | None
    period_days: float | None
    depth_ppm: float | None
    fpp: float | None
    pathway: str | None
    top_flags: tuple[str, ...]   # up to 3 most important flag strings
    formatted_card: str
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _extract_fpp(d: dict) -> float | None:
    """Extract FPP from various dict layouts."""
    for key in ("false_positive_probability", "fpp", "best_fpp"):
        v = d.get(key)
        if v is not None:
            return float(v)
    scores = d.get("scores", {})
    if isinstance(scores, dict):
        v = scores.get("false_positive_probability")
        if v is not None:
            return float(v)
    return None


def _extract_flags(d: dict) -> list[str]:
    """Extract top candidate flags from explanation or vetting keys."""
    flags: list[str] = []
    # Check explanation dict
    explanation = d.get("explanation", {})
    if isinstance(explanation, dict):
        for key in ("blocking_evidence", "negative_evidence"):
            val = explanation.get(key)
            if isinstance(val, (list, tuple)):
                flags.extend(str(v) for v in val)
    # Check top-level flag fields
    for key in ("flags", "vetting_flags"):
        val = d.get(key)
        if isinstance(val, (list, tuple)):
            flags.extend(str(v) for v in val)
    return flags[:3]


def build_summary_card(candidate_dict: dict) -> SummaryCardResult:
    """Build a compact summary card from a scored candidate dict.

    Args:
        candidate_dict: Output row from ``exo --output`` or ``batch_scan``.

    Returns:
        :class:`SummaryCardResult`.
    """
    if not isinstance(candidate_dict, dict):
        return SummaryCardResult(None, None, None, None, None, (), "", "INVALID")
    if not candidate_dict:
        return SummaryCardResult(None, None, None, None, None, (), "", "EMPTY")

    tic_id = str(candidate_dict.get("tic_id", "—"))
    period = candidate_dict.get("period_days")
    depth = candidate_dict.get("depth_ppm")
    fpp = _extract_fpp(candidate_dict)
    pathway = candidate_dict.get("pathway") or candidate_dict.get("submission_pathway")
    top_flags = tuple(_extract_flags(candidate_dict))

    # Build card text
    lines = [
        f"**TIC {tic_id}**",
        f"Period: {f'{period:.4f} d' if period is not None else '—'}  |  "
        f"Depth: {f'{depth:.0f} ppm' if depth is not None else '—'}  |  "
        f"FPP: {f'{fpp:.3f}' if fpp is not None else '—'}",
        f"Pathway: {pathway or '—'}",
    ]
    if top_flags:
        lines.append("Flags: " + " · ".join(top_flags[:3]))

    card = "\n".join(lines)

    return SummaryCardResult(
        tic_id=tic_id,
        period_days=period,
        depth_ppm=depth,
        fpp=fpp,
        pathway=pathway,
        top_flags=top_flags,
        formatted_card=card,
        flag="OK",
    )


def format_summary_card(result: SummaryCardResult) -> str:
    """Format summary card result as Markdown."""
    if result.flag in ("EMPTY", "INVALID"):
        return f"## Candidate Summary Card\n\n_Flag: {result.flag}_\n"
    return f"## Candidate Summary Card\n\n{result.formatted_card}\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="candidate_summary_card",
        description="Generate a compact summary card from a candidate JSON dict.",
    )
    parser.add_argument("--json", type=str, default=None, help="Candidate JSON string")
    args = parser.parse_args(argv)

    d = json.loads(args.json) if args.json else {}
    result = build_summary_card(d)
    print(format_summary_card(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
