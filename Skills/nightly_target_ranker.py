"""Rank targets for a given observing night by composite score.

Scoring components:
  - transit_priority: 1 if a transit occurs tonight, 0 otherwise
  - airmass_score: 1 - (airmass - 1) / 4  (clamped to [0,1]; airmass=1 → 1.0)
  - magnitude_score: 1 - (mag - mag_min) / (mag_max - mag_min)  (clamped to [0,1])

Composite = 0.50 * transit_priority + 0.30 * airmass_score + 0.20 * magnitude_score

Public API
----------
NightlyRankResult(ranked_targets, n_targets, best_target, flag)
rank_nightly_targets(targets, *, night_date) -> NightlyRankResult
format_nightly_rank_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NightlyRankResult:
    ranked_targets: tuple[dict, ...]
    n_targets: int
    best_target: str
    flag: str  # "OK" | "NO_TARGETS" | "ALL_SAME_SCORE"


def _airmass_score(airmass: float) -> float:
    """Convert airmass to [0,1] score (lower airmass → higher score)."""
    return max(0.0, min(1.0, 1.0 - (airmass - 1.0) / 4.0))


def _magnitude_score(mag: float, mag_min: float, mag_max: float) -> float:
    """Brightness score; brighter (lower mag) → higher score."""
    if mag_max <= mag_min:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (mag - mag_min) / (mag_max - mag_min)))


def rank_nightly_targets(
    targets: list[dict],
    *,
    night_date: str = "",  # informational
) -> NightlyRankResult:
    """Rank targets for tonight.

    Each target dict should have:
        ``name`` (str), ``airmass`` (float), ``mag`` (float),
        ``transit_tonight`` (bool, default False).

    Args:
        targets: List of target dicts.
        night_date: Date string (informational only).

    Returns:
        :class:`NightlyRankResult`.
    """
    _ = night_date  # informational

    if not targets:
        return NightlyRankResult(
            ranked_targets=(),
            n_targets=0,
            best_target="",
            flag="NO_TARGETS",
        )

    mags = [t.get("mag", 12.0) for t in targets]
    mag_min, mag_max = min(mags), max(mags)

    scored = []
    for t in targets:
        airmass = float(t.get("airmass", 2.0))
        mag = float(t.get("mag", 12.0))
        transit_p = 1.0 if t.get("transit_tonight", False) else 0.0
        a_score = _airmass_score(airmass)
        m_score = _magnitude_score(mag, mag_min, mag_max)
        composite = round(0.50 * transit_p + 0.30 * a_score + 0.20 * m_score, 4)
        scored.append({**t, "composite_score": composite})

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(scored, 1):
        s["rank"] = i

    scores = [s["composite_score"] for s in scored]
    all_same = len(set(scores)) == 1 and len(scores) > 1

    best = scored[0].get("name", "") if scored else ""
    flag = "ALL_SAME_SCORE" if all_same else "OK"

    return NightlyRankResult(
        ranked_targets=tuple(scored),
        n_targets=len(scored),
        best_target=str(best),
        flag=flag,
    )


def format_nightly_rank_result(result: NightlyRankResult) -> str:
    """Format nightly rank result as Markdown."""
    lines = [
        "## Nightly Target Ranking",
        "",
        f"- Targets: {result.n_targets}",
        f"- Best target: {result.best_target}",
        f"- **Flag: {result.flag}**",
    ]
    if result.ranked_targets:
        lines += [
            "",
            "| Rank | Name | Airmass | Mag | Transit | Score |",
            "|------|------|---------|-----|---------|-------|",
        ]
        for t in result.ranked_targets:
            transit_yn = "Yes" if t.get("transit_tonight") else "No"
            lines.append(
                f"| {t['rank']} | {t.get('name', '?')} |"
                f" {t.get('airmass', '?')} | {t.get('mag', '?')} |"
                f" {transit_yn} | {t['composite_score']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="nightly_target_ranker",
        description="Rank targets for a given observing night.",
    )
    parser.add_argument("targets_json", type=str, help="JSON file with list of target dicts")
    parser.add_argument("--night-date", type=str, default="")
    args = parser.parse_args(argv)

    with open(args.targets_json) as fh:
        targets = json.load(fh)

    result = rank_nightly_targets(targets, night_date=args.night_date)
    print(format_nightly_rank_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
