"""Summarize the orbital architecture of a multi-planet system.

Given a list of planet dicts with keys: period_days, radius_rearth,
mass_mearth (optional), compute:
- architecture_class: "compact" if all P<50d, "spread" if any P>200d,
  "mixed" otherwise; "SINGLE" for 1-planet systems
- Period ratios between adjacent planets sorted by period
- Counts by radius class: n_rocky (R<2), n_super_earth (2<=R<4),
  n_neptune (4<=R<8), n_giant (R>=8)

Public API
----------
SystemArchResult(n_planets, architecture_class, period_ratio_min,
                 period_ratio_max, n_rocky, n_giant, flag)
summarize_system_architecture(planets) -> SystemArchResult
format_system_arch(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemArchResult:
    n_planets: int
    architecture_class: str  # "compact" / "spread" / "mixed" / "SINGLE"
    period_ratio_min: float | None   # min ratio between adjacent pairs
    period_ratio_max: float | None   # max ratio between adjacent pairs
    n_rocky: int        # R < 2
    n_giant: int        # R >= 8
    flag: str           # "OK", "SINGLE", "COMPACT"


def summarize_system_architecture(planets: list[dict]) -> SystemArchResult:
    """Summarize multi-planet system orbital architecture.

    Args:
        planets: List of planet dicts, each with at minimum 'period_days'
                 and 'radius_rearth' keys.

    Returns:
        :class:`SystemArchResult`.
    """
    if not planets:
        return SystemArchResult(
            n_planets=0,
            architecture_class="EMPTY",
            period_ratio_min=None,
            period_ratio_max=None,
            n_rocky=0,
            n_giant=0,
            flag="ERROR",
        )

    n = len(planets)

    # Sort by period
    sorted_planets = sorted(planets, key=lambda p: float(p.get("period_days", 0.0)))

    # Count by radius class
    n_rocky = 0
    n_giant = 0
    for p in sorted_planets:
        r = float(p.get("radius_rearth", 1.0))
        if r < 2.0:
            n_rocky += 1
        if r >= 8.0:
            n_giant += 1

    # Period ratios between adjacent pairs
    periods = [float(p.get("period_days", 0.0)) for p in sorted_planets]

    if n == 1:
        arch_class = "SINGLE"
        ratio_min = None
        ratio_max = None
        flag = "SINGLE"
    else:
        ratios: list[float] = []
        for i in range(1, len(periods)):
            if periods[i - 1] > 0:
                ratios.append(periods[i] / periods[i - 1])
        ratio_min = round(min(ratios), 4) if ratios else None
        ratio_max = round(max(ratios), 4) if ratios else None

        # Architecture classification
        all_compact = all(p <= 50.0 for p in periods)
        any_spread = any(p > 200.0 for p in periods)

        if all_compact:
            arch_class = "compact"
            flag = "COMPACT"
        elif any_spread:
            arch_class = "spread"
            flag = "OK"
        else:
            arch_class = "mixed"
            flag = "OK"

    return SystemArchResult(
        n_planets=n,
        architecture_class=arch_class,
        period_ratio_min=ratio_min,
        period_ratio_max=ratio_max,
        n_rocky=n_rocky,
        n_giant=n_giant,
        flag=flag,
    )


def format_system_arch(result: SystemArchResult) -> str:
    """Format system architecture summary as Markdown."""
    ratio_str = (
        f"{result.period_ratio_min:.3f} – {result.period_ratio_max:.3f}"
        if result.period_ratio_min is not None
        else "N/A"
    )
    lines = [
        "## System Architecture",
        "",
        f"- Planets: **{result.n_planets}**",
        f"- Architecture: **{result.architecture_class}**",
        f"- Period ratio range: {ratio_str}",
        f"- Rocky planets (R<2 R⊕): {result.n_rocky}",
        f"- Giant planets (R≥8 R⊕): {result.n_giant}",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="system_architecture_summarizer",
        description=__doc__,
    )
    p.add_argument(
        "--planets",
        type=str,
        default="[]",
        help='JSON list of planet dicts with period_days and radius_rearth keys',
    )
    args = p.parse_args(argv)
    planets = json.loads(args.planets)
    r = summarize_system_architecture(planets)
    print(format_system_arch(r))
    return 0 if r.flag not in ("ERROR",) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
