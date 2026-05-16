"""Rank a list of TIC IDs by scan priority, incorporating TOI status and sector coverage.

Composes the TOI check, sector coverage query, and a priority heuristic into a
single ranked recommendation list.  All external dependencies are injectable so
the module works in tests without network access.

Public API
----------
prioritize_targets(tic_ids, *, toi_table_fn, sector_coverage_fn, priority_fn,
                   min_priority, skip_known_tois) -> list[TargetRecommendation]
format_recommendations(recs) -> str
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TargetRecommendation:
    tic_id: int
    priority_score: float
    toi_status: dict | None        # from check_toi(); None = not a TOI
    n_sectors: int | None          # from get_sector_coverage(); None = not queried
    recommendation: str            # "scan", "skip_toi", "skip_low_priority"
    reason: str


# ---------------------------------------------------------------------------
# Internal defaults
# ---------------------------------------------------------------------------


def _default_toi_check(tic_id: int, toi_table_fn: Callable[[], str] | None) -> dict | None:
    """Call toi_checker.check_toi with the given table function."""
    try:
        from Skills.toi_checker import check_toi  # noqa: PLC0415
    except ImportError:
        return None
    return check_toi(tic_id, toi_table_fn=toi_table_fn)


def _default_priority_fn(tic_id: int, n_sectors: int | None) -> float:
    """Simple heuristic when no star-scanner priority function is provided."""
    return 0.5 + (n_sectors or 0) * 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prioritize_targets(
    tic_ids: list[int],
    *,
    toi_check_fn: Callable[..., dict | None] | None = None,
    toi_table_fn: Callable[[], str] | None = None,
    sector_coverage_fn: Callable[[str], Any] | None = None,
    priority_fn: Callable[..., float] | None = None,
    min_priority: float = 0.30,
    skip_known_tois: bool = True,
) -> list[TargetRecommendation]:
    """Rank TIC IDs by scan priority.

    Args:
        tic_ids: List of TESS Input Catalog identifiers.
        toi_check_fn: Injectable TOI lookup callable ``(tic_id, toi_table_fn) -> dict|None``.
            Defaults to ``Skills.toi_checker.check_toi``.
        toi_table_fn: Passed through to ``toi_check_fn``; defaults to live ExoFOP fetch.
        sector_coverage_fn: Callable accepting a target string and returning an
            object with an ``n_sectors`` attribute (or integer).  If ``None``,
            sector count is left as ``None``.
        priority_fn: Callable accepting ``(tic_id, n_sectors)`` and returning a
            float in [0, 1].  Defaults to the built-in heuristic.
        min_priority: Targets with priority below this threshold are labelled
            "skip_low_priority".
        skip_known_tois: If ``True``, targets found in the TOI list are labelled
            "skip_toi" regardless of priority.

    Returns:
        Sorted list of :class:`TargetRecommendation` objects, highest priority first.
    """
    recs: list[TargetRecommendation] = []
    _toi_fn = toi_check_fn if toi_check_fn is not None else _default_toi_check

    for tic_id in tic_ids:
        # TOI check
        toi_status = _toi_fn(tic_id, toi_table_fn)

        # Sector coverage
        n_sectors: int | None = None
        if sector_coverage_fn is not None:
            raw = sector_coverage_fn(f"TIC {tic_id}")
            if hasattr(raw, "n_sectors"):
                n_sectors = raw.n_sectors
            elif isinstance(raw, int):
                n_sectors = raw
            else:
                n_sectors = None

        # Priority
        calc_priority = priority_fn if priority_fn is not None else _default_priority_fn
        score = float(calc_priority(tic_id, n_sectors))

        # Recommendation logic
        if isinstance(toi_status, dict) and skip_known_tois:
            recommendation = "skip_toi"
            toi_id = toi_status.get('toi', '?')
            disposition = toi_status.get('disposition', '?')
            reason = f"Already in TOI list as {toi_id} ({disposition})"
        elif score < min_priority:
            recommendation = "skip_low_priority"
            reason = f"Priority score {score:.3f} below threshold {min_priority}"
        else:
            recommendation = "scan"
            reason = f"Priority score {score:.3f} meets threshold"

        recs.append(TargetRecommendation(
            tic_id=tic_id,
            priority_score=score,
            toi_status=toi_status,
            n_sectors=n_sectors,
            recommendation=recommendation,
            reason=reason,
        ))

    recs.sort(key=lambda r: r.priority_score, reverse=True)
    return recs


def format_recommendations(recs: list[TargetRecommendation]) -> str:
    """Return a Markdown table summarising target recommendations.

    Args:
        recs: List returned by :func:`prioritize_targets`.

    Returns:
        Markdown string, or ``"_No targets._\\n"`` for an empty list.
    """
    if not recs:
        return "_No targets._\n"

    header = "| TIC ID | Priority | Sectors | Recommendation | Reason |"
    sep    = "| --- | --- | --- | --- | --- |"
    lines  = [header, sep]
    for r in recs:
        sectors = str(r.n_sectors) if r.n_sectors is not None else "—"
        lines.append(
            f"| {r.tic_id} | {r.priority_score:.3f} | {sectors} "
            f"| {r.recommendation} | {r.reason} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="target_prioritizer",
        description="Rank TIC IDs by scan priority using TOI + sector + heuristic score.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        metavar="FILE",
        help="Text file with one TIC ID per line.",
    )
    parser.add_argument(
        "--skip-tois",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip targets already in the ExoFOP TOI list (default: --skip-tois).",
    )
    parser.add_argument(
        "--min-priority",
        type=float,
        default=0.30,
        metavar="X",
        help="Minimum priority score (default: 0.30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write recommendations as JSON to this file.",
    )
    args = parser.parse_args(argv)

    raw_ids = [
        line.strip()
        for line in Path(args.input_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    tic_ids = [int(x) for x in raw_ids]

    recs = prioritize_targets(
        tic_ids,
        min_priority=args.min_priority,
        skip_known_tois=args.skip_tois,
    )

    if args.output:
        import dataclasses  # noqa: PLC0415
        args.output.write_text(
            json.dumps([dataclasses.asdict(r) for r in recs], indent=2)
        )
        print(f"Recommendations written to {args.output}")
    else:
        print(format_recommendations(recs))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
