"""Propagate labels from confirmed planets to period harmonics.

When a planet at period P is confirmed, the same TIC ID at P/2, 2P, P/3, 3P
etc. is almost certainly also a planet (or at least not a false positive).
Propagating labels to harmonics can increase the training corpus size without
requiring additional observations.

Public API
----------
PropagationResult(original_rows, propagated_rows, n_added, harmonic_factors,
                  flag)
propagate_labels(rows, *, pos_label, harmonic_factors, period_rtol) -> PropagationResult
format_propagation_report(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationResult:
    original_rows: tuple[dict, ...]
    propagated_rows: tuple[dict, ...]   # new rows only
    n_added: int
    harmonic_factors: tuple[float, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def propagate_labels(
    rows: list[dict],
    *,
    pos_label: str = "planet_candidate",
    harmonic_factors: tuple[float, ...] = (0.5, 2.0, 3.0),
    period_rtol: float = 0.01,
) -> PropagationResult:
    """Generate harmonic-period entries for positive-class rows.

    For each positive row with period P, creates new rows at each harmonic
    factor (P*f) unless a row for that TIC + period already exists.

    Args:
        rows: Input label rows with ``tic_id``, ``period_days``, ``label``.
        pos_label: Label value for the positive class.
        harmonic_factors: Multipliers applied to P to form harmonic periods.
        period_rtol: Relative tolerance for existing-period check.

    Returns:
        PropagationResult with new propagated rows (original rows not modified).
    """
    if not isinstance(rows, list):
        return PropagationResult(original_rows=(), propagated_rows=(), n_added=0,
                                 harmonic_factors=harmonic_factors, flag="INVALID")
    if not rows:
        return PropagationResult(original_rows=(), propagated_rows=(), n_added=0,
                                 harmonic_factors=harmonic_factors, flag="EMPTY")

    # Build existing (tic_id, period) index for duplicate detection
    existing: dict[int, list[float]] = {}
    for r in rows:
        try:
            tic_id = int(r["tic_id"])
            period = float(r["period_days"])
            existing.setdefault(tic_id, []).append(period)
        except (KeyError, TypeError, ValueError):
            pass

    def _period_exists(tic_id: int, period: float) -> bool:
        for p in existing.get(tic_id, []):
            denom = max(abs(p), abs(period), 1e-12)
            if abs(p - period) / denom < period_rtol:
                return True
        return False

    propagated: list[dict] = []
    for r in rows:
        if r.get("label") != pos_label:
            continue
        try:
            tic_id = int(r["tic_id"])
            period = float(r["period_days"])
        except (KeyError, TypeError, ValueError):
            continue

        for factor in harmonic_factors:
            new_period = period * factor
            if new_period <= 0:
                continue
            if _period_exists(tic_id, new_period):
                continue
            new_row = dict(r)
            new_row["period_days"] = new_period
            new_row["source"] = f"propagated_{r.get('source', 'unknown')}_x{factor}"
            propagated.append(new_row)
            existing.setdefault(tic_id, []).append(new_period)

    return PropagationResult(
        original_rows=tuple(rows),
        propagated_rows=tuple(propagated),
        n_added=len(propagated),
        harmonic_factors=harmonic_factors,
        flag="OK",
    )


def format_propagation_report(result: PropagationResult) -> str:
    """Format a Markdown label propagation report.

    Args:
        result: PropagationResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Label Propagation Report\n",
        f"Flag: `{result.flag}`\n",
    ]
    if result.flag in ("EMPTY", "INVALID"):
        lines.append(f"\n_{result.flag}: no data to propagate._\n")
        return "\n".join(lines)

    factors_str = ", ".join(str(f) for f in result.harmonic_factors)
    lines += [
        f"**Harmonic factors**: {factors_str}\n",
        f"**Original rows**: {len(result.original_rows)}\n",
        f"**Propagated rows added**: {result.n_added}\n",
        f"**Total after merge**: {len(result.original_rows) + result.n_added}\n",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Propagate labels to period harmonics.")
    parser.add_argument("label_json", help="Input JSON label rows.")
    parser.add_argument("--output", help="Write merged rows to this file.")
    parser.add_argument("--factors", nargs="+", type=float, default=[0.5, 2.0, 3.0])
    parser.add_argument("--pos-label", default="planet_candidate")
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.label_json).read_text())
    result = propagate_labels(rows, pos_label=args.pos_label,
                              harmonic_factors=tuple(args.factors))
    print(format_propagation_report(result))

    if args.output:
        merged = list(result.original_rows) + list(result.propagated_rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(merged, indent=2))
        print(f"\nWrote {len(merged)} rows to {args.output}")

    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
