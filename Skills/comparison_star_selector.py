"""Select optimal comparison stars for ground-based differential photometry."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonStar:
    star_id: str
    magnitude: float
    color_index: float | None
    separation_arcsec: float
    magnitude_diff: float      # |mag - target_mag|
    color_diff: float | None   # |color - target_color|
    composite_score: float     # lower = better
    recommended: bool


@dataclass(frozen=True)
class ComparisonSelectionResult:
    n_candidates: int
    n_recommended: int
    target_magnitude: float
    stars: tuple[ComparisonStar, ...]
    flag: str


def select_comparison_stars(
    target_magnitude: float,
    candidate_stars: list[dict],
    target_color: float | None = None,
    max_mag_diff: float = 1.0,
    max_separation_arcsec: float = 600.0,
    min_separation_arcsec: float = 5.0,
    n_recommend: int = 5,
) -> ComparisonSelectionResult:
    """
    Rank and select comparison stars for differential photometry.

    Composite score = mag_diff + 0.5 * color_diff + 0.01 * (separation/60)
    Lower score = better comparison star.

    Parameters
    ----------
    target_magnitude:       Target star magnitude (V or T).
    candidate_stars:        List of dicts with keys: star_id, magnitude,
                            color_index (optional), separation_arcsec.
    target_color:           Target star color index (optional).
    max_mag_diff:           Maximum magnitude difference to consider.
    max_separation_arcsec:  Maximum angular separation from target.
    min_separation_arcsec:  Minimum separation (avoid blending).
    n_recommend:            Number of stars to mark as recommended.
    """
    if not math.isfinite(target_magnitude):
        return ComparisonSelectionResult(
            n_candidates=0, n_recommended=0,
            target_magnitude=target_magnitude, stars=(), flag="INVALID_TARGET_MAG",
        )
    if not candidate_stars:
        return ComparisonSelectionResult(
            n_candidates=0, n_recommended=0,
            target_magnitude=target_magnitude, stars=(), flag="NO_CANDIDATES",
        )

    scored: list[ComparisonStar] = []
    for s in candidate_stars:
        try:
            mag = float(s.get("magnitude", float("nan")))
            sep = float(s.get("separation_arcsec", float("nan")))
            star_id = str(s.get("star_id", "unknown"))
            color = s.get("color_index")
            color = float(color) if color is not None else None
        except (TypeError, ValueError):
            continue

        if not math.isfinite(mag) or not math.isfinite(sep):
            continue
        if sep < min_separation_arcsec or sep > max_separation_arcsec:
            continue

        mag_diff = abs(mag - target_magnitude)
        if mag_diff > max_mag_diff:
            continue

        if color is not None and target_color is not None and math.isfinite(color):
            color_diff: float | None = abs(color - target_color)
            score = mag_diff + 0.5 * color_diff + 0.01 * (sep / 60.0)
        else:
            color_diff = None
            score = mag_diff + 0.01 * (sep / 60.0)

        scored.append(ComparisonStar(
            star_id=star_id,
            magnitude=mag,
            color_index=color,
            separation_arcsec=sep,
            magnitude_diff=round(mag_diff, 4),
            color_diff=round(color_diff, 4) if color_diff is not None else None,
            composite_score=round(score, 4),
            recommended=False,
        ))

    scored.sort(key=lambda s: s.composite_score)

    # Mark top n_recommend as recommended
    final: list[ComparisonStar] = []
    for i, s in enumerate(scored):
        final.append(ComparisonStar(
            star_id=s.star_id, magnitude=s.magnitude, color_index=s.color_index,
            separation_arcsec=s.separation_arcsec, magnitude_diff=s.magnitude_diff,
            color_diff=s.color_diff, composite_score=s.composite_score,
            recommended=(i < n_recommend),
        ))

    n_rec = min(n_recommend, len(final))
    return ComparisonSelectionResult(
        n_candidates=len(final),
        n_recommended=n_rec,
        target_magnitude=target_magnitude,
        stars=tuple(final),
        flag="OK",
    )


def format_comparison_result(r: ComparisonSelectionResult) -> str:
    if r.flag != "OK":
        return f"No result (flag: {r.flag}).\n"
    lines = [
        f"**Comparison Star Selection** — {r.n_candidates} candidates, "
        f"{r.n_recommended} recommended\n",
        "| Star ID | Mag | Sep (\") | Mag diff | Score | Rec |",
        "|---|---|---|---|---|---|",
    ]
    for s in r.stars:
        lines.append(
            f"| {s.star_id} | {s.magnitude:.2f} | {s.separation_arcsec:.1f} | "
            f"{s.magnitude_diff:.3f} | {s.composite_score:.3f} | "
            f"{'Y' if s.recommended else 'n'} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Select comparison stars for photometry.")
    p.add_argument("target_magnitude", type=float)
    p.add_argument("--max-mag-diff", type=float, default=1.0)
    p.add_argument("--n-recommend", type=int, default=5)
    args = p.parse_args()
    import json
    import sys
    data = json.load(sys.stdin)
    r = select_comparison_stars(
        args.target_magnitude, data,
        max_mag_diff=args.max_mag_diff, n_recommend=args.n_recommend,
    )
    print(format_comparison_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
