"""Detect whether a candidate is likely a duplicate of a known TOI.

Compares a candidate's TIC ID, period, and epoch against the ExoFOP TOI list
and returns the closest match with similarity scores.

Public API
----------
DuplicateMatch(toi, tic_id, period_days, epoch_bjd, period_diff_days,
               epoch_diff_days, is_duplicate, similarity)
DuplicateDetectionResult(input_tic_id, input_period, input_epoch,
                         matches, is_duplicate, best_match, flag)
detect_duplicate_toi(tic_id, period_days, epoch_bjd, *, toi_rows,
                     period_tol_days, epoch_tol_days) -> DuplicateDetectionResult
format_duplicate_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicateMatch:
    toi: str
    tic_id: int
    period_days: float
    epoch_bjd: float
    period_diff_days: float
    epoch_diff_days: float
    is_duplicate: bool
    similarity: float          # 0.0–1.0


@dataclass(frozen=True)
class DuplicateDetectionResult:
    input_tic_id: int
    input_period: float
    input_epoch: float
    matches: tuple[DuplicateMatch, ...]
    is_duplicate: bool
    best_match: DuplicateMatch | None
    flag: str                  # "DUPLICATE", "POSSIBLE_DUPLICATE", "UNIQUE", "NO_DATA"


def _period_similarity(diff: float, tol: float) -> float:
    """Similarity score falling from 1.0 (exact) to 0.0 (at tol)."""
    return max(0.0, 1.0 - diff / tol)


def detect_duplicate_toi(
    tic_id: int,
    period_days: float,
    epoch_bjd: float,
    *,
    toi_rows: list[dict],
    period_tol_days: float = 0.01,
    epoch_tol_days: float = 0.05,
    duplicate_threshold: float = 0.9,
    possible_threshold: float = 0.5,
) -> DuplicateDetectionResult:
    """Check if (tic_id, period, epoch) matches a known TOI.

    Args:
        tic_id: TIC ID of the candidate.
        period_days: Candidate orbital period in days.
        epoch_bjd: Candidate transit epoch (BJD).
        toi_rows: List of dicts with keys ``toi``, ``tic_id``, ``period_days``,
            ``epoch_bjd``.  These are the rows from the ExoFOP TOI table.
        period_tol_days: Period tolerance for matching.
        epoch_tol_days: Epoch tolerance (mod period) for matching.
        duplicate_threshold: Similarity above which flag = DUPLICATE.
        possible_threshold: Similarity above which flag = POSSIBLE_DUPLICATE.

    Returns:
        :class:`DuplicateDetectionResult`.
    """
    if not toi_rows:
        return DuplicateDetectionResult(
            tic_id, period_days, epoch_bjd, (), False, None, "NO_DATA"
        )

    matches: list[DuplicateMatch] = []

    for row in toi_rows:
        row_tic = int(row.get("tic_id", -1))
        row_period = float(row.get("period_days", 0.0))
        row_epoch = float(row.get("epoch_bjd", 0.0))
        row_toi = str(row.get("toi", ""))

        if row_tic != tic_id:
            continue
        if row_period <= 0:
            continue

        # Period matching (also check half/double period aliases)
        period_diff = abs(period_days - row_period)
        half_diff = abs(period_days - row_period / 2.0)
        double_diff = abs(period_days - row_period * 2.0)
        best_period_diff = min(period_diff, half_diff, double_diff)

        # Epoch match modulo period
        if row_period > 0:
            epoch_diff_raw = abs(epoch_bjd - row_epoch) % row_period
            epoch_diff = min(epoch_diff_raw, row_period - epoch_diff_raw)
        else:
            epoch_diff = abs(epoch_bjd - row_epoch)

        p_sim = _period_similarity(best_period_diff, period_tol_days)
        e_sim = _period_similarity(epoch_diff, epoch_tol_days)
        similarity = (p_sim + e_sim) / 2.0

        is_dup = similarity >= duplicate_threshold

        matches.append(DuplicateMatch(
            toi=row_toi,
            tic_id=row_tic,
            period_days=row_period,
            epoch_bjd=row_epoch,
            period_diff_days=round(best_period_diff, 6),
            epoch_diff_days=round(epoch_diff, 6),
            is_duplicate=is_dup,
            similarity=round(similarity, 4),
        ))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    best = matches[0] if matches else None

    if best is None:
        is_dup = False
        flag = "UNIQUE"
    elif best.similarity >= duplicate_threshold:
        is_dup = True
        flag = "DUPLICATE"
    elif best.similarity >= possible_threshold:
        is_dup = False
        flag = "POSSIBLE_DUPLICATE"
    else:
        is_dup = False
        flag = "UNIQUE"

    return DuplicateDetectionResult(
        input_tic_id=tic_id,
        input_period=period_days,
        input_epoch=epoch_bjd,
        matches=tuple(matches),
        is_duplicate=is_dup,
        best_match=best,
        flag=flag,
    )


def format_duplicate_result(result: DuplicateDetectionResult) -> str:
    """Format duplicate detection result as Markdown."""
    lines = [
        "## Duplicate TOI Detection",
        "",
        f"- TIC ID: {result.input_tic_id}",
        f"- Input period: {result.input_period:.6f} d",
        f"- Input epoch: {result.input_epoch:.4f} BJD",
        f"- Flag: **{result.flag}**",
    ]
    if result.best_match:
        m = result.best_match
        lines += [
            "",
            f"### Best Match: TOI {m.toi}",
            f"- TOI period: {m.period_days:.6f} d (Δ = {m.period_diff_days:.6f} d)",
            f"- TOI epoch: {m.epoch_bjd:.4f} BJD (Δ = {m.epoch_diff_days:.6f} d)",
            f"- Similarity score: {m.similarity:.3f}",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="duplicate_toi_detector",
        description="Check if a candidate duplicates a known TOI.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--toi-table", required=True, metavar="JSON")
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.toi_table).read_text())
    result = detect_duplicate_toi(args.tic_id, args.period_days, args.epoch_bjd, toi_rows=rows)
    print(format_duplicate_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
