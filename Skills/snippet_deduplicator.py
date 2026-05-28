"""Remove duplicate phase-folded snippets from the CNN training corpus.

Two snippets are considered duplicates when they share the same TIC ID and
have periods within a fractional tolerance (default 1%). Keeps the first
occurrence in input order.

Public API
----------
DeduplicationResult(n_input, n_output, n_removed, duplicate_tic_ids, flag)
deduplicate_snippets(rows, *, period_rtol) -> DeduplicationResult
format_dedup_report(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeduplicationResult:
    n_input: int
    n_output: int
    n_removed: int
    duplicate_tic_ids: tuple[int, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def deduplicate_snippets(
    rows: list[dict],
    *,
    period_rtol: float = 0.01,
) -> DeduplicationResult:
    """Remove duplicate snippets from a label/snippet row list.

    Args:
        rows: List of dicts with at least ``tic_id`` and ``period_days``.
        period_rtol: Relative tolerance for period matching; two periods
            p1, p2 are considered equal when |p1-p2|/max(p1,p2) < period_rtol.

    Returns:
        DeduplicationResult; ``rows`` is not mutated.
    """
    if not isinstance(rows, list):
        return DeduplicationResult(n_input=0, n_output=0, n_removed=0,
                                   duplicate_tic_ids=(), flag="INVALID")
    if not rows:
        return DeduplicationResult(n_input=0, n_output=0, n_removed=0,
                                   duplicate_tic_ids=(), flag="EMPTY")

    seen: dict[int, list[float]] = {}   # tic_id -> list of kept periods
    kept_indices: list[int] = []
    dup_tic_ids: set[int] = set()

    for idx, row in enumerate(rows):
        try:
            tic_id = int(row["tic_id"])
            period = float(row.get("period_days") or 0.0)
        except (KeyError, TypeError, ValueError):
            kept_indices.append(idx)
            continue

        existing = seen.get(tic_id, [])
        is_dup = False
        for p in existing:
            denom = max(abs(p), abs(period), 1e-12)
            if abs(p - period) / denom < period_rtol:
                is_dup = True
                dup_tic_ids.add(tic_id)
                break

        if not is_dup:
            seen.setdefault(tic_id, []).append(period)
            kept_indices.append(idx)

    n_out = len(kept_indices)
    n_removed = len(rows) - n_out

    return DeduplicationResult(
        n_input=len(rows),
        n_output=n_out,
        n_removed=n_removed,
        duplicate_tic_ids=tuple(sorted(dup_tic_ids)),
        flag="OK",
    )


def apply_deduplication(rows: list[dict], *, period_rtol: float = 0.01) -> list[dict]:
    """Return deduplicated rows (convenience wrapper).

    Args:
        rows: Input rows.
        period_rtol: Period match tolerance.

    Returns:
        Deduplicated list (new list, original untouched).
    """
    result = deduplicate_snippets(rows, period_rtol=period_rtol)
    # Rebuild from scratch using same logic to return actual rows
    seen: dict[int, list[float]] = {}
    output: list[dict] = []
    for row in rows:
        try:
            tic_id = int(row["tic_id"])
            period = float(row.get("period_days") or 0.0)
        except (KeyError, TypeError, ValueError):
            output.append(row)
            continue

        existing = seen.get(tic_id, [])
        is_dup = any(
            abs(p - period) / max(abs(p), abs(period), 1e-12) < period_rtol
            for p in existing
        )
        if not is_dup:
            seen.setdefault(tic_id, []).append(period)
            output.append(row)
    _ = result  # result stats already computed
    return output


def format_dedup_report(result: DeduplicationResult) -> str:
    """Format a Markdown deduplication report.

    Args:
        result: DeduplicationResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Snippet Deduplication Report\n",
        f"Flag: `{result.flag}` | Input: {result.n_input} | Output: {result.n_output}\n",
    ]
    if result.flag in ("EMPTY", "INVALID"):
        lines.append(f"\n_{result.flag}: no data to deduplicate._\n")
        return "\n".join(lines)

    pct = 100.0 * result.n_removed / result.n_input if result.n_input else 0.0
    lines.append(f"**Removed**: {result.n_removed} duplicates ({pct:.1f}%)\n")
    lines.append(f"**Affected TIC IDs**: {len(result.duplicate_tic_ids)}\n")
    if result.duplicate_tic_ids:
        sample = list(result.duplicate_tic_ids[:5])
        more = len(result.duplicate_tic_ids) - 5
        s = ", ".join(str(t) for t in sample)
        if more > 0:
            s += f" … (+{more} more)"
        lines.append(f"Sample: {s}\n")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Deduplicate snippet/label rows.")
    parser.add_argument("input_json", help="Input JSON rows file.")
    parser.add_argument("--output", help="Write deduplicated rows to this file.")
    parser.add_argument("--period-rtol", type=float, default=0.01)
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.input_json).read_text())
    result = deduplicate_snippets(rows, period_rtol=args.period_rtol)
    print(format_dedup_report(result))

    if args.output:
        out_rows = apply_deduplication(rows, period_rtol=args.period_rtol)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out_rows, indent=2))
        print(f"\nWrote {len(out_rows)} rows to {args.output}")

    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
