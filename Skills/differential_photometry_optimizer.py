"""Optimize a comparison star ensemble for differential photometry.

Selects comparison stars within max_delta_mag of target, not variable, and
in-field. The ensemble precision scales as 1/sqrt(n_selected).

Public API
----------
DiffPhotResult(selected_indices, ensemble_mag, expected_precision_ppm, n_selected, flag)
optimize_diff_photometry(target_mag, comparison_stars, *, max_delta_mag, min_stars)
    -> DiffPhotResult
format_diff_phot(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffPhotResult:
    selected_indices: tuple[int, ...]
    ensemble_mag: float
    expected_precision_ppm: float
    n_selected: int
    flag: str  # "OK", "FEW_STARS", "NO_STARS"


def optimize_diff_photometry(
    target_mag: float,
    comparison_stars: list[dict],
    *,
    max_delta_mag: float = 1.5,
    min_stars: int = 3,
) -> DiffPhotResult:
    """Select comparison stars for differential photometry.

    Args:
        target_mag: Target magnitude.
        comparison_stars: List of dicts with keys ``mag`` (float),
            ``variable`` (bool), ``in_field`` (bool).
        max_delta_mag: Maximum magnitude difference from target.
        min_stars: Minimum number of comparison stars for "OK" flag.

    Returns:
        :class:`DiffPhotResult`.
    """
    selected = []
    for i, star in enumerate(comparison_stars):
        mag = float(star.get("mag", 99.0))
        variable = bool(star.get("variable", False))
        in_field = bool(star.get("in_field", True))
        if (
            abs(mag - target_mag) <= max_delta_mag
            and not variable
            and in_field
        ):
            selected.append((i, mag))

    if not selected:
        return DiffPhotResult(
            selected_indices=(),
            ensemble_mag=target_mag,
            expected_precision_ppm=1e6,
            n_selected=0,
            flag="NO_STARS",
        )

    indices = tuple(i for i, _ in selected)
    mags = [m for _, m in selected]
    # Ensemble magnitude: average in flux space
    flux_sum = sum(10.0 ** (-m / 2.5) for m in mags)
    ensemble_mag = -2.5 * math.log10(flux_sum / len(mags)) if flux_sum > 0 else 99.0

    # Simplified precision: photon-noise limited, scaled by sqrt(n)
    # Assume target at ~1e5 electrons per image; precision ~ 1/sqrt(n_ens)
    base_precision_ppm = 1000.0  # ~1 mmag per star
    expected_precision_ppm = base_precision_ppm / math.sqrt(len(selected))

    n_sel = len(selected)
    flag = "OK" if n_sel >= min_stars else "FEW_STARS"

    return DiffPhotResult(
        selected_indices=indices,
        ensemble_mag=round(ensemble_mag, 3),
        expected_precision_ppm=round(expected_precision_ppm, 2),
        n_selected=n_sel,
        flag=flag,
    )


def format_diff_phot(result: DiffPhotResult) -> str:
    """Format differential photometry result as Markdown."""
    lines = [
        "## Differential Photometry Optimizer",
        "",
        f"- Stars selected: {result.n_selected}",
        f"- Selected indices: {list(result.selected_indices)}",
        f"- Ensemble magnitude: {result.ensemble_mag:.3f}",
        f"- Expected precision: {result.expected_precision_ppm:.1f} ppm",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-mag", type=float, default=12.0)
    p.add_argument("--stars-json", help="JSON file with list of star dicts")
    p.add_argument("--max-delta-mag", type=float, default=1.5)
    p.add_argument("--min-stars", type=int, default=3)
    args = p.parse_args(argv)

    stars: list[dict] = []
    if args.stars_json:
        with open(args.stars_json) as fh:
            stars = json.load(fh)

    r = optimize_diff_photometry(
        args.target_mag, stars, max_delta_mag=args.max_delta_mag, min_stars=args.min_stars
    )
    print(format_diff_phot(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
