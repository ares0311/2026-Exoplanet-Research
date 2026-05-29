"""Select the best guide star from a catalog of nearby stars.

Criteria: magnitude within [guide_mag_min, guide_mag_max], separation in
[min_sep_arcsec, max_sep_arcsec].  The best guide star minimises |mag - optimal|
where optimal is the midpoint of the guide magnitude range.

Public API
----------
GuideStarResult(n_candidates, selected_index, selected_mag, selected_sep_arcsec, flag)
select_guide_star(stars, *, guide_mag_min, guide_mag_max, min_sep_arcsec, max_sep_arcsec)
    -> GuideStarResult
format_guide_star(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideStarResult:
    n_candidates: int
    selected_index: int | None
    selected_mag: float | None
    selected_sep_arcsec: float | None
    flag: str  # "OK", "NO_GUIDE_STAR", "TOO_FEW_CANDIDATES"


def select_guide_star(
    stars: list[dict],
    *,
    guide_mag_min: float = 8.0,
    guide_mag_max: float = 14.0,
    min_sep_arcsec: float = 30.0,
    max_sep_arcsec: float = 600.0,
) -> GuideStarResult:
    """Select best guide star from a catalog.

    Args:
        stars: List of dicts with keys ``mag`` (float) and ``sep_arcsec`` (float).
        guide_mag_min: Minimum guide star magnitude (bright limit).
        guide_mag_max: Maximum guide star magnitude (faint limit).
        min_sep_arcsec: Minimum separation from target in arcseconds.
        max_sep_arcsec: Maximum separation from target in arcseconds.

    Returns:
        :class:`GuideStarResult`.
    """
    if not stars:
        return GuideStarResult(
            n_candidates=0,
            selected_index=None,
            selected_mag=None,
            selected_sep_arcsec=None,
            flag="NO_GUIDE_STAR",
        )

    optimal_mag = (guide_mag_min + guide_mag_max) / 2.0

    candidates = []
    for i, s in enumerate(stars):
        mag = float(s.get("mag", 99.0))
        sep = float(s.get("sep_arcsec", 0.0))
        if (
            guide_mag_min <= mag <= guide_mag_max
            and min_sep_arcsec <= sep <= max_sep_arcsec
        ):
            score = -abs(mag - optimal_mag)  # higher = closer to optimal mag
            candidates.append((score, i, mag, sep))

    if not candidates:
        return GuideStarResult(
            n_candidates=0,
            selected_index=None,
            selected_mag=None,
            selected_sep_arcsec=None,
            flag="NO_GUIDE_STAR",
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    flag = "TOO_FEW_CANDIDATES" if len(candidates) < 2 else "OK"

    return GuideStarResult(
        n_candidates=len(candidates),
        selected_index=best[1],
        selected_mag=round(best[2], 3),
        selected_sep_arcsec=round(best[3], 2),
        flag=flag,
    )


def format_guide_star(result: GuideStarResult) -> str:
    """Format guide star selection as Markdown."""
    lines = [
        "## Guide Star Selection",
        "",
        f"- Candidates found: {result.n_candidates}",
    ]
    if result.selected_index is not None:
        lines += [
            f"- Selected index: {result.selected_index}",
            f"- Selected mag: {result.selected_mag:.3f}",
            f"- Separation: {result.selected_sep_arcsec:.2f} arcsec",
        ]
    else:
        lines.append("- No suitable guide star found.")
    lines.append(f"- **Flag: {result.flag}**")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stars_json", help="JSON file with list of star dicts")
    p.add_argument("--guide-mag-min", type=float, default=8.0)
    p.add_argument("--guide-mag-max", type=float, default=14.0)
    p.add_argument("--min-sep-arcsec", type=float, default=30.0)
    p.add_argument("--max-sep-arcsec", type=float, default=600.0)
    args = p.parse_args(argv)
    with open(args.stars_json) as fh:
        stars = json.load(fh)
    r = select_guide_star(
        stars,
        guide_mag_min=args.guide_mag_min,
        guide_mag_max=args.guide_mag_max,
        min_sep_arcsec=args.min_sep_arcsec,
        max_sep_arcsec=args.max_sep_arcsec,
    )
    print(format_guide_star(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
