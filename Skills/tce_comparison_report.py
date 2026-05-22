"""Cross-match a candidate against a TCE table and flag agreements/conflicts.

Given a candidate's period and epoch, searches a table of Threshold Crossing
Events (TCEs) for matching entries.  A match is defined by period agreement
within a fractional tolerance and epoch agreement within half a transit
duration.

Public API
----------
TCEMatch(tce_id, tce_period_days, tce_epoch, period_delta_frac,
          epoch_delta_hours, match_type)
TCEComparisonResult(tic_id, n_tces_checked, n_matches, matches,
                    best_match, flag)
compare_tce(tic_id, period_days, epoch_bjd, tce_table, *,
            period_tol_frac, epoch_tol_hours,
            duration_hours) -> TCEComparisonResult
format_tce_comparison(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCEMatch:
    tce_id: str
    tce_period_days: float
    tce_epoch: float
    period_delta_frac: float   # |P_cand - P_tce| / P_cand
    epoch_delta_hours: float   # |T0_cand - T0_tce| mod P in hours
    match_type: str            # "exact" | "alias" | "harmonic" | "conflict"


@dataclass(frozen=True)
class TCEComparisonResult:
    tic_id: int
    n_tces_checked: int
    n_matches: int
    matches: tuple[TCEMatch, ...]
    best_match: TCEMatch | None
    flag: str  # "OK" | "NO_MATCH" | "NO_TCES" | "INVALID"


def _epoch_delta_hours(t0_cand: float, t0_tce: float, period: float) -> float:
    """Minimum epoch difference modulo period, in hours."""
    diff = abs(t0_cand - t0_tce) % period
    if diff > period / 2:
        diff = period - diff
    return diff * 24.0


def compare_tce(
    tic_id: int,
    period_days: float,
    epoch_bjd: float,
    tce_table: list[dict],
    *,
    period_tol_frac: float = 0.02,
    epoch_tol_hours: float = 1.0,
    duration_hours: float = 2.0,
) -> TCEComparisonResult:
    """Compare a candidate against a TCE table.

    Args:
        tic_id: Target TIC ID.
        period_days: Candidate period in days.
        epoch_bjd: Candidate mid-transit epoch (BJD).
        tce_table: List of dicts, each with keys ``tic_id`` (int),
            ``tce_id`` (str), ``period_days`` (float), ``epoch`` (float).
            Rows with non-matching ``tic_id`` are skipped.
        period_tol_frac: Fractional period tolerance for match.
        epoch_tol_hours: Epoch tolerance in hours.
        duration_hours: Transit duration (used for epoch tolerance default).

    Returns:
        :class:`TCEComparisonResult`.
    """
    if period_days <= 0:
        return TCEComparisonResult(tic_id, 0, 0, (), None, "INVALID")

    # Filter table for this TIC
    rows = [r for r in tce_table if r.get("tic_id") == tic_id]
    if not rows:
        return TCEComparisonResult(tic_id, 0, 0, (), None, "NO_TCES")

    matches: list[TCEMatch] = []
    p_tol_hrs = max(epoch_tol_hours, duration_hours / 2.0)

    for row in rows:
        tce_p = float(row.get("period_days", 0.0))
        tce_ep = float(row.get("epoch", 0.0))
        tce_id_str = str(row.get("tce_id", ""))

        if tce_p <= 0:
            continue

        # Check direct period match
        p_delta = abs(period_days - tce_p) / period_days
        ep_delta = _epoch_delta_hours(epoch_bjd, tce_ep, period_days)

        if p_delta <= period_tol_frac and ep_delta <= p_tol_hrs:
            mtype = "exact"
        else:
            # Check 2:1 alias
            alias_p = tce_p * 2.0
            alias_delta = abs(period_days - alias_p) / period_days
            half_p = tce_p / 2.0
            half_delta = abs(period_days - half_p) / period_days
            if min(alias_delta, half_delta) <= period_tol_frac:
                mtype = "alias"
                p_delta = min(alias_delta, half_delta)
                ep_delta = _epoch_delta_hours(epoch_bjd, tce_ep, period_days)
            elif p_delta > 0.2:
                continue  # not related
            else:
                mtype = "conflict"

        matches.append(TCEMatch(
            tce_id=tce_id_str,
            tce_period_days=tce_p,
            tce_epoch=tce_ep,
            period_delta_frac=round(p_delta, 5),
            epoch_delta_hours=round(ep_delta, 3),
            match_type=mtype,
        ))

    if not matches:
        return TCEComparisonResult(tic_id, len(rows), 0, (), None, "NO_MATCH")

    # Best match: prefer exact > alias > conflict, then smallest period delta
    _prio = {"exact": 0, "alias": 1, "harmonic": 2, "conflict": 3}
    best = min(matches, key=lambda m: (_prio.get(m.match_type, 9), m.period_delta_frac))

    return TCEComparisonResult(
        tic_id=tic_id,
        n_tces_checked=len(rows),
        n_matches=len(matches),
        matches=tuple(matches),
        best_match=best,
        flag="OK",
    )


def format_tce_comparison(result: TCEComparisonResult) -> str:
    """Format TCE comparison result as Markdown."""
    lines = [
        "## TCE Comparison Report",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- TCEs checked: {result.n_tces_checked}",
        f"- Matches: {result.n_matches}",
        f"- **Flag: {result.flag}**",
    ]
    if result.best_match:
        m = result.best_match
        lines += [
            "",
            f"**Best match**: TCE `{m.tce_id}` — {m.match_type}",
            f"  Period: {m.tce_period_days:.4f} d (Δ={m.period_delta_frac:.4f})",
            f"  Epoch Δ: {m.epoch_delta_hours:.2f} h",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tce_comparison_report",
        description="Compare a candidate against a TCE table.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    args = parser.parse_args(argv)

    result = compare_tce(args.tic_id, args.period_days, args.epoch_bjd, [])
    print(format_tce_comparison(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
