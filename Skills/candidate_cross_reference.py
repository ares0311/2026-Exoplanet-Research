"""Cross-reference a transit candidate against a catalog of known objects.

Matches by TIC ID, then by period proximity for independent period check.

Public API
----------
CrossRefMatch(tic_id, period_days, catalog_name, catalog_id,
              period_match, period_delta_frac, disposition, note)
CrossRefResult(tic_id, period_days, matches, n_matches, best_match,
               period_confirmed, flag)
cross_reference(tic_id, period_days, catalog_rows, *, period_rtol,
                fetch_fn) -> CrossRefResult
format_cross_ref_result(result) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CrossRefMatch:
    tic_id: int
    period_days: float | None
    catalog_name: str
    catalog_id: str
    period_match: bool
    period_delta_frac: float | None  # |Δperiod|/period_ref; None if no period available
    disposition: str
    note: str


@dataclass(frozen=True)
class CrossRefResult:
    tic_id: int
    period_days: float | None
    matches: tuple[CrossRefMatch, ...]
    n_matches: int
    best_match: CrossRefMatch | None
    period_confirmed: bool  # True if any match has period_match=True
    flag: str  # "KNOWN" | "PERIOD_CONFLICT" | "NOT_FOUND" | "INVALID"


def _frac_diff(a: float, b: float) -> float:
    ref = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / ref


def cross_reference(
    tic_id: int,
    period_days: float | None,
    catalog_rows: list[dict],
    *,
    period_rtol: float = 0.01,
    catalog_name: str = "catalog",
) -> CrossRefResult:
    """Cross-reference a candidate against a list of catalog rows.

    Args:
        tic_id: TIC identifier of the candidate.
        period_days: Orbital period for period matching.
        catalog_rows: List of dicts with keys: tic_id, period_days (opt),
                      catalog_id (opt), disposition (opt), note (opt).
        period_rtol: Relative tolerance for period matching.
        catalog_name: Name of the catalog for labeling matches.

    Returns:
        CrossRefResult with all matches and overall flag.
    """
    if not isinstance(catalog_rows, list):
        return CrossRefResult(
            tic_id=tic_id,
            period_days=period_days,
            matches=(),
            n_matches=0,
            best_match=None,
            period_confirmed=False,
            flag="INVALID",
        )

    matches: list[CrossRefMatch] = []
    for row in catalog_rows:
        row_tic: int | None = None
        with contextlib.suppress(TypeError, ValueError):
            row_tic = int(row.get("tic_id") or 0) or None

        if row_tic != tic_id:
            continue

        cat_period: float | None = None
        with contextlib.suppress(TypeError, ValueError):
            raw = row.get("period_days")
            if raw is not None:
                cat_period = float(raw)

        period_match = False
        delta_frac: float | None = None
        if period_days is not None and cat_period is not None:
            delta_frac = round(_frac_diff(period_days, cat_period), 6)
            period_match = delta_frac < period_rtol

        matches.append(CrossRefMatch(
            tic_id=tic_id,
            period_days=cat_period,
            catalog_name=catalog_name,
            catalog_id=str(row.get("catalog_id") or row.get("toi") or ""),
            period_match=period_match,
            period_delta_frac=delta_frac,
            disposition=str(row.get("disposition") or ""),
            note=str(row.get("note") or ""),
        ))

    period_confirmed = any(m.period_match for m in matches)
    best_match: CrossRefMatch | None = None
    if matches:
        best_match = min(
            matches,
            key=lambda m: (m.period_delta_frac if m.period_delta_frac is not None else 1.0),
        )

    if not matches:
        flag = "NOT_FOUND"
    elif period_days is not None and matches and not period_confirmed:
        flag = "PERIOD_CONFLICT"
    else:
        flag = "KNOWN"

    return CrossRefResult(
        tic_id=tic_id,
        period_days=period_days,
        matches=tuple(matches),
        n_matches=len(matches),
        best_match=best_match,
        period_confirmed=period_confirmed,
        flag=flag,
    )


def format_cross_ref_result(result: CrossRefResult) -> str:
    """Format cross-reference result as Markdown.

    Args:
        result: CrossRefResult to format.

    Returns:
        Markdown string.
    """
    period_str = (f"{result.period_days:.4f} d"
                  if result.period_days is not None else "unknown")
    lines = [
        f"## Cross-Reference — TIC {result.tic_id}\n",
        f"**Query period**: {period_str} | "
        f"**Matches**: {result.n_matches} | **Status**: `{result.flag}`\n",
    ]
    if not result.matches:
        lines.append("\n_No catalog matches found._")
        return "\n".join(lines)

    lines += [
        "",
        "| Catalog | ID | Period (d) | Δperiod | Disposition | Period Match |",
        "|---|---|---|---|---|---|",
    ]
    for m in result.matches:
        p_str = f"{m.period_days:.4f}" if m.period_days is not None else "—"
        delta_str = f"{m.period_delta_frac:.4f}" if m.period_delta_frac is not None else "—"
        match_str = "✓" if m.period_match else "✗"
        lines.append(
            f"| {m.catalog_name} | {m.catalog_id} | {p_str} | {delta_str} | "
            f"{m.disposition} | {match_str} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Cross-reference a candidate.")
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--catalog", required=True, help="Catalog JSON path.")
    parser.add_argument("--period-rtol", type=float, default=0.01)
    args = parser.parse_args(argv)

    from pathlib import Path
    rows = json.loads(Path(args.catalog).read_text())
    result = cross_reference(args.tic_id, args.period, rows,
                             period_rtol=args.period_rtol)
    print(format_cross_ref_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
