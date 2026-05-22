"""Jackknife sector test for signal persistence.

Checks whether a periodic signal is driven by one particular sector or is
consistently present across all available sectors.  Each sector is left out
in turn and the remaining data are folded and checked for a transit-like dip.
If the signal disappears when a single sector is removed it is flagged as
sector-dependent (possible artifact); if it persists in all leave-one-out
subsets it is flagged as persistent.

Public API
----------
SectorPersistenceResult(n_sectors, n_persistent, persistence_fraction,
                        sector_ids, depths_per_sector,
                        is_persistent, flag)
check_signal_persistence(time, flux, period_days, epoch_bjd, sector_ids, *,
                         duration_hours, depth_threshold_frac,
                         min_sectors) -> SectorPersistenceResult
format_persistence_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorPersistenceResult:
    n_sectors: int
    n_persistent: int                  # sectors where signal detected in leave-one-out
    persistence_fraction: float
    sector_ids: tuple[int, ...]
    depths_per_sector: tuple[float | None, ...]  # per-sector in-transit depth
    is_persistent: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def _sector_depth(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    half_width: float,
) -> float | None:
    """Estimate in-transit depth relative to OOT mean for a single sector."""
    phases = _phase_fold(time, epoch, period)
    in_flux: list[float] = []
    oot_flux: list[float] = []
    oot_half = min(3 * half_width, 0.4)
    for ph, f in zip(phases, flux, strict=False):
        ap = abs(ph)
        if ap <= half_width:
            in_flux.append(f)
        elif half_width < ap <= oot_half:
            oot_flux.append(f)
    if len(in_flux) < 2 or len(oot_flux) < 2:
        return None
    baseline = sum(oot_flux) / len(oot_flux)
    in_mean = sum(in_flux) / len(in_flux)
    return baseline - in_mean  # positive = dip


def check_signal_persistence(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    sector_ids: list[int],
    *,
    duration_hours: float = 2.0,
    depth_threshold_frac: float = 0.5,
    min_sectors: int = 2,
    persistence_threshold: float = 0.7,
) -> SectorPersistenceResult:
    """Run a jackknife sector test for signal persistence.

    Args:
        time: Time array (same units as epoch_bjd, e.g. BJD).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch.
        sector_ids: Integer sector label for each cadence (same length as time).
        duration_hours: Transit duration in hours.
        depth_threshold_frac: A sector counts as persistent if the leave-one-out
            depth is at least this fraction of the full-sample depth.
        min_sectors: Minimum number of sectors to produce an OK result.
        persistence_threshold: Minimum persistence fraction for ``is_persistent``.

    Returns:
        :class:`SectorPersistenceResult`.
    """
    n = len(flux)
    if n < 10 or len(time) != n or len(sector_ids) != n or period_days <= 0:
        return SectorPersistenceResult(0, 0, 0.0, (), (), False, "INVALID")

    unique_sectors = sorted(set(sector_ids))
    n_sec = len(unique_sectors)

    if n_sec < min_sectors:
        return SectorPersistenceResult(
            n_sec, 0, 0.0, tuple(unique_sectors), (), False, "INSUFFICIENT"
        )

    half_width = (duration_hours / 24.0) / period_days / 2.0

    # Full-sample depth (reference)
    full_depth = _sector_depth(time, flux, period_days, epoch_bjd, half_width)
    if full_depth is None or full_depth <= 0:
        return SectorPersistenceResult(
            n_sec, 0, 0.0, tuple(unique_sectors), (), False, "INSUFFICIENT"
        )

    depths: list[float | None] = []
    n_pers = 0

    for excl in unique_sectors:
        t_lo = [time[i] for i in range(n) if sector_ids[i] != excl]
        f_lo = [flux[i] for i in range(n) if sector_ids[i] != excl]
        d = _sector_depth(t_lo, f_lo, period_days, epoch_bjd, half_width)
        depths.append(d)
        if d is not None and d >= depth_threshold_frac * full_depth:
            n_pers += 1

    pf = n_pers / n_sec
    is_pers = pf >= persistence_threshold

    return SectorPersistenceResult(
        n_sectors=n_sec,
        n_persistent=n_pers,
        persistence_fraction=round(pf, 4),
        sector_ids=tuple(unique_sectors),
        depths_per_sector=tuple(
            round(d, 6) if d is not None else None for d in depths
        ),
        is_persistent=is_pers,
        flag="OK",
    )


def format_persistence_result(result: SectorPersistenceResult) -> str:
    """Format signal persistence result as Markdown."""
    lines = [
        "## Signal Persistence Check",
        "",
        f"- Sectors: {result.n_sectors}",
        f"- Persistent in: {result.n_persistent}",
        f"- Persistence fraction: {result.persistence_fraction:.2%}",
        f"- Persistent: {'Yes' if result.is_persistent else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    if result.sector_ids:
        lines += ["", "| Sector | Depth (leave-one-out) |", "|---|---|"]
        for sid, d in zip(result.sector_ids, result.depths_per_sector, strict=False):
            lines.append(f"| {sid} | {d} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="signal_persistence_checker",
        description="Jackknife sector test for signal persistence.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = check_signal_persistence(
        [], [], args.period_days, args.epoch_bjd, [],
        duration_hours=args.duration_hours,
    )
    print(format_persistence_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
