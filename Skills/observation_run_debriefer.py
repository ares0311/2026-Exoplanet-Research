"""Debrief a ground-based observation run and score its quality.

Quality score:
  quality_score = (n_usable/n_images)*0.4 + (1 - cloud_cover_pct/100)*0.3
                  + (1 - min(seeing/3.0, 1))*0.2 + (1 - min(airmass_max/2.5, 1))*0.1

Grades: EXCELLENT > 0.85, GOOD > 0.65, MARGINAL > 0.40, POOR otherwise.

Public API
----------
RunDebrief(n_usable, quality_score, issues, recommendations, flag)
debrief_run(n_images, n_rejected, airmass_max, seeing_arcsec, cloud_cover_pct,
            issues_list) -> RunDebrief
format_run_debrief(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunDebrief:
    n_usable: int
    quality_score: float  # [0, 1]
    issues: tuple[str, ...]
    recommendations: tuple[str, ...]
    flag: str  # "EXCELLENT", "GOOD", "MARGINAL", "POOR"


def debrief_run(
    n_images: int,
    n_rejected: int,
    airmass_max: float = 1.5,
    seeing_arcsec: float = 2.0,
    cloud_cover_pct: float = 0.0,
    issues_list: list[str] | None = None,
) -> RunDebrief:
    """Debrief an observation run.

    Args:
        n_images: Total images taken.
        n_rejected: Images rejected (cosmic rays, clouds, guiding failures, etc.).
        airmass_max: Maximum airmass during the run.
        seeing_arcsec: Median seeing in arcseconds.
        cloud_cover_pct: Cloud cover percentage during the run.
        issues_list: Optional list of issue strings observed during the run.

    Returns:
        :class:`RunDebrief`.
    """
    n_usable = max(0, n_images - n_rejected)
    usable_frac = n_usable / max(n_images, 1)

    quality_score = (
        usable_frac * 0.4
        + (1.0 - cloud_cover_pct / 100.0) * 0.3
        + (1.0 - min(seeing_arcsec / 3.0, 1.0)) * 0.2
        + (1.0 - min(airmass_max / 2.5, 1.0)) * 0.1
    )
    quality_score = max(0.0, min(1.0, quality_score))

    issues: list[str] = list(issues_list) if issues_list else []

    # Auto-add issues from metrics
    if cloud_cover_pct > 30:
        issues.append(f"Significant cloud cover ({cloud_cover_pct:.0f}%)")
    if seeing_arcsec > 3.0:
        issues.append(f"Poor seeing ({seeing_arcsec:.1f} arcsec)")
    if airmass_max > 2.0:
        issues.append(f"High airmass reached ({airmass_max:.2f})")
    if usable_frac < 0.5:
        issues.append(f"Low usable fraction ({usable_frac:.0%})")

    # Recommendations
    recommendations: list[str] = []
    if cloud_cover_pct > 0:
        recommendations.append("Check light curves for systematic steps due to cloud passages")
    if seeing_arcsec > 2.5:
        recommendations.append("Consider larger aperture radius in reduction")
    if airmass_max > 1.8:
        recommendations.append("Apply differential colour extinction correction")
    if usable_frac < 0.8:
        recommendations.append("Inspect rejected frames for guiding failures")
    if not recommendations:
        recommendations.append("No corrective action needed; reduction can proceed normally")

    if quality_score > 0.85:
        flag = "EXCELLENT"
    elif quality_score > 0.65:
        flag = "GOOD"
    elif quality_score > 0.40:
        flag = "MARGINAL"
    else:
        flag = "POOR"

    return RunDebrief(
        n_usable=n_usable,
        quality_score=round(quality_score, 4),
        issues=tuple(issues),
        recommendations=tuple(recommendations),
        flag=flag,
    )


def format_run_debrief(result: RunDebrief) -> str:
    """Format run debrief as Markdown."""
    lines = [
        "## Observation Run Debrief",
        "",
        f"- Usable images: {result.n_usable}",
        f"- Quality score: {result.quality_score:.4f}",
        f"- **Flag: {result.flag}**",
    ]
    if result.issues:
        lines += ["", "### Issues"]
        for issue in result.issues:
            lines.append(f"- {issue}")
    if result.recommendations:
        lines += ["", "### Recommendations"]
        for rec in result.recommendations:
            lines.append(f"- {rec}")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--n-rejected", type=int, default=10)
    p.add_argument("--airmass-max", type=float, default=1.5)
    p.add_argument("--seeing", type=float, default=2.0)
    p.add_argument("--cloud-cover-pct", type=float, default=0.0)
    args = p.parse_args(argv)
    r = debrief_run(
        args.n_images, args.n_rejected, args.airmass_max, args.seeing, args.cloud_cover_pct
    )
    print(format_run_debrief(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
