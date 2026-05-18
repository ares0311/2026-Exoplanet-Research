"""Match a candidate's period and epoch against a known reference catalog.

Compares period (within a fractional tolerance) and phase offset (within a
time tolerance) to determine whether a candidate matches a known planet,
eclipsing binary, or other object.

Public API
----------
GroundTruthEntry(name, period_days, epoch_bjd, category)
GroundTruthMatchResult(tic_id, candidate_period, candidate_epoch,
                       matched_name, matched_category, period_ratio,
                       epoch_offset_days, flag)
match_ground_truth(tic_id, period_days, epoch_bjd, catalog, *,
                   period_tol_frac, epoch_tol_days) -> GroundTruthMatchResult
format_match_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruthEntry:
    name: str
    period_days: float
    epoch_bjd: float
    category: str  # "planet", "eb", "variable", "unknown"


@dataclass(frozen=True)
class GroundTruthMatchResult:
    tic_id: int
    candidate_period: float
    candidate_epoch: float
    matched_name: str | None
    matched_category: str | None
    period_ratio: float | None
    epoch_offset_days: float | None
    flag: str  # "MATCH", "NO_MATCH", "EMPTY_CATALOG"


def _period_match(p_cand: float, p_ref: float, tol_frac: float) -> tuple[bool, float]:
    """Return (matched, ratio) for integer harmonic period matching."""
    best_ratio = p_cand / p_ref
    for num in range(1, 4):
        for den in range(1, 4):
            ratio = num / den
            if abs(p_cand - p_ref * ratio) / p_ref < tol_frac:
                return True, p_cand / p_ref
    return False, best_ratio


def match_ground_truth(
    tic_id: int,
    period_days: float,
    epoch_bjd: float,
    catalog: list[GroundTruthEntry],
    *,
    period_tol_frac: float = 0.02,
    epoch_tol_days: float = 0.5,
) -> GroundTruthMatchResult:
    """Match a candidate against a ground-truth catalog.

    Args:
        tic_id: TESS Input Catalog ID.
        period_days: Candidate period in days.
        epoch_bjd: Candidate epoch in BJD.
        catalog: List of :class:`GroundTruthEntry` objects.
        period_tol_frac: Fractional period tolerance (default 2%).
        epoch_tol_days: Maximum allowed epoch offset in days.

    Returns:
        :class:`GroundTruthMatchResult`.
    """
    if not catalog:
        return GroundTruthMatchResult(
            tic_id, period_days, epoch_bjd,
            None, None, None, None, "EMPTY_CATALOG",
        )

    for entry in catalog:
        if entry.period_days <= 0:
            continue
        matched, ratio = _period_match(period_days, entry.period_days, period_tol_frac)
        if not matched:
            continue
        # Phase offset: how far is candidate epoch from nearest reference transit?
        n = round((epoch_bjd - entry.epoch_bjd) / entry.period_days)
        predicted_epoch = entry.epoch_bjd + n * entry.period_days
        epoch_offset = abs(epoch_bjd - predicted_epoch)
        if epoch_offset > epoch_tol_days:
            continue
        return GroundTruthMatchResult(
            tic_id=tic_id,
            candidate_period=period_days,
            candidate_epoch=epoch_bjd,
            matched_name=entry.name,
            matched_category=entry.category,
            period_ratio=round(ratio, 6),
            epoch_offset_days=round(epoch_offset, 4),
            flag="MATCH",
        )

    return GroundTruthMatchResult(
        tic_id=tic_id,
        candidate_period=period_days,
        candidate_epoch=epoch_bjd,
        matched_name=None,
        matched_category=None,
        period_ratio=None,
        epoch_offset_days=None,
        flag="NO_MATCH",
    )


def format_match_result(result: GroundTruthMatchResult) -> str:
    """Format ground truth match result as Markdown."""
    lines = [
        "## Ground Truth Match",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- Candidate period: {result.candidate_period:.4f} days",
        f"- Candidate epoch: {result.candidate_epoch:.4f} BJD",
    ]
    if result.flag == "EMPTY_CATALOG":
        lines.append("- **Flag: EMPTY_CATALOG** — no catalog provided")
    elif result.flag == "MATCH":
        lines += [
            f"- Matched: **{result.matched_name}** ({result.matched_category})",
            f"- Period ratio: {result.period_ratio:.6f}",
            f"- Epoch offset: {result.epoch_offset_days:.4f} days",
            "- **Flag: MATCH**",
        ]
    else:
        lines.append("- No catalog match found")
        lines.append("- **Flag: NO_MATCH**")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ground_truth_matcher",
        description="Match a candidate against a known catalog.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--period-tol-frac", type=float, default=0.02)
    parser.add_argument("--epoch-tol-days", type=float, default=0.5)
    args = parser.parse_args(argv)

    result = match_ground_truth(
        args.tic_id, args.period_days, args.epoch_bjd, [],
        period_tol_frac=args.period_tol_frac,
        epoch_tol_days=args.epoch_tol_days,
    )
    print(format_match_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
