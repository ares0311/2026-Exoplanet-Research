"""Normalise per-sector flux baselines to a common reference level.

When stacking multiple TESS sectors the median flux level can differ between
sectors due to scattered-light trends or momentum-dump artefacts.  This module
re-normalises each sector to a common baseline (default: grand median of all
sectors combined) using either additive offset or multiplicative scaling.

Public API
----------
SectorNormResult(sector_id, n_points, offset, scale, method)
BaselineNormResult(n_sectors, sector_results, normalized_flux, flag)
normalize_sector_baselines(time, flux, sector_ids, *,
                           method, reference) -> BaselineNormResult
format_baseline_norm_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorNormResult:
    sector_id: int
    n_points: int
    offset: float          # additive shift applied (0 for multiplicative)
    scale: float           # multiplicative factor applied (1 for additive)
    method: str            # "additive" | "multiplicative"


@dataclass(frozen=True)
class BaselineNormResult:
    n_sectors: int
    sector_results: tuple[SectorNormResult, ...]
    normalized_flux: tuple[float, ...]
    flag: str  # "OK" | "SINGLE_SECTOR" | "INVALID"


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def normalize_sector_baselines(
    time: list[float],
    flux: list[float],
    sector_ids: list[int],
    *,
    method: str = "additive",
    reference: float | None = None,
) -> BaselineNormResult:
    """Normalise per-sector flux baselines.

    Args:
        time: Time array (not used for computation, validated for length).
        flux: Flux array (normalised around ~1 or counts).
        sector_ids: Integer sector ID per cadence, same length as flux.
        method: ``"additive"`` (subtract per-sector median, add reference) or
            ``"multiplicative"`` (divide by per-sector median, multiply by reference).
        reference: Target baseline level.  Defaults to grand median.

    Returns:
        :class:`BaselineNormResult`.
    """
    n = len(flux)
    if n < 2 or len(sector_ids) != n or len(time) != n:
        return BaselineNormResult(0, (), (), "INVALID")
    if method not in ("additive", "multiplicative"):
        return BaselineNormResult(0, (), (), "INVALID")

    # Group indices by sector
    sectors: dict[int, list[int]] = {}
    for i, sid in enumerate(sector_ids):
        sectors.setdefault(sid, []).append(i)

    if reference is None:
        reference = _median(list(flux))

    normalized = list(flux)
    sector_results: list[SectorNormResult] = []

    for sid in sorted(sectors):
        idxs = sectors[sid]
        sector_flux = [flux[i] for i in idxs]
        med = _median(sector_flux)

        if method == "additive":
            offset = reference - med
            scale = 1.0
            for i in idxs:
                normalized[i] = flux[i] + offset
        else:
            scale = reference / med if abs(med) > 1e-20 else 1.0
            offset = 0.0
            for i in idxs:
                normalized[i] = flux[i] * scale

        sector_results.append(SectorNormResult(
            sector_id=sid,
            n_points=len(idxs),
            offset=round(offset, 8),
            scale=round(scale, 8),
            method=method,
        ))

    flag = "SINGLE_SECTOR" if len(sectors) == 1 else "OK"
    return BaselineNormResult(
        n_sectors=len(sectors),
        sector_results=tuple(sector_results),
        normalized_flux=tuple(round(f, 8) for f in normalized),
        flag=flag,
    )


def format_baseline_norm_result(result: BaselineNormResult) -> str:
    """Format baseline normalisation result as Markdown."""
    lines = [
        "## Sector Baseline Normalisation",
        "",
        f"- Sectors normalised: {result.n_sectors}",
        f"- **Flag: {result.flag}**",
    ]
    if result.sector_results:
        lines.append("")
        lines.append("| Sector | N points | Offset | Scale |")
        lines.append("|---|---|---|---|")
        for sr in result.sector_results:
            lines.append(f"| {sr.sector_id} | {sr.n_points} | {sr.offset:.6f} | {sr.scale:.6f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="sector_baseline_normalizer",
        description="Normalise per-sector flux baselines.",
    )
    parser.add_argument("--method", choices=["additive", "multiplicative"], default="additive")
    args = parser.parse_args(argv)

    result = normalize_sector_baselines([], [], [], method=args.method)
    print(format_baseline_norm_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
