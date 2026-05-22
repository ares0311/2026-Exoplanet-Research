"""Track out-of-transit RMS sector-by-sector to flag systematic noise spikes.

Splits the light curve by sector, masks in-transit cadences, and computes the
RMS of the remaining out-of-transit (OOT) flux for each sector.  A sector
whose OOT RMS is significantly higher than the median across all sectors is
flagged as potentially affected by systematics.

Public API
----------
SectorRMS(sector_id, n_oot_cadences, rms, is_elevated, z_score)
OOTRMSResult(n_sectors, sector_rms, median_rms, n_elevated, flag)
track_oot_rms(time, flux, period_days, epoch_bjd, sector_ids, *,
              duration_hours, sigma_threshold) -> OOTRMSResult
format_oot_rms_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorRMS:
    sector_id: int
    n_oot_cadences: int
    rms: float
    is_elevated: bool
    z_score: float | None


@dataclass(frozen=True)
class OOTRMSResult:
    n_sectors: int
    sector_rms: tuple[SectorRMS, ...]
    median_rms: float | None
    n_elevated: int
    flag: str  # "OK" | "ALL_ELEVATED" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def track_oot_rms(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    sector_ids: list[int],
    *,
    duration_hours: float = 2.0,
    sigma_threshold: float = 2.5,
) -> OOTRMSResult:
    """Compute per-sector OOT RMS and flag elevated sectors.

    Args:
        time: Time array (BJD or same units as epoch_bjd).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch.
        sector_ids: Integer sector label per cadence.
        duration_hours: Transit duration in hours (used to define OOT window).
        sigma_threshold: Sectors whose RMS exceeds median + threshold * MAD
            are flagged as elevated.

    Returns:
        :class:`OOTRMSResult`.
    """
    n = len(flux)
    if n < 5 or len(time) != n or len(sector_ids) != n or period_days <= 0:
        return OOTRMSResult(0, (), None, 0, "INVALID")

    half_width = (duration_hours / 24.0) / period_days / 2.0
    phases = _phase_fold(time, epoch_bjd, period_days)

    unique = sorted(set(sector_ids))
    sector_rms_list: list[SectorRMS] = []
    rms_values: list[float] = []

    for sec in unique:
        oot_flux: list[float] = []
        for i, sid in enumerate(sector_ids):
            if sid == sec and abs(phases[i]) > half_width:
                oot_flux.append(flux[i])
        if len(oot_flux) < 3:
            continue
        mean = sum(oot_flux) / len(oot_flux)
        rms = math.sqrt(sum((f - mean) ** 2 for f in oot_flux) / len(oot_flux))
        sector_rms_list.append(SectorRMS(sec, len(oot_flux), round(rms, 8), False, None))
        rms_values.append(rms)

    if not rms_values:
        return OOTRMSResult(0, (), None, 0, "INSUFFICIENT")

    sorted_rms = sorted(rms_values)
    n_r = len(sorted_rms)
    if n_r % 2 == 1:
        median = sorted_rms[n_r // 2]
    else:
        median = (sorted_rms[n_r // 2 - 1] + sorted_rms[n_r // 2]) / 2.0
    devs = sorted(abs(r - median) for r in rms_values)
    if n_r % 2 == 1:
        mad = devs[n_r // 2] * 1.4826
    else:
        mad = (devs[n_r // 2 - 1] + devs[n_r // 2]) / 2.0 * 1.4826
    threshold = median + sigma_threshold * max(mad, 1e-30)

    updated: list[SectorRMS] = []
    n_elev = 0
    for sr in sector_rms_list:
        z = (sr.rms - median) / max(mad, 1e-30) if mad > 0 else None
        elevated = sr.rms > threshold
        if elevated:
            n_elev += 1
        updated.append(SectorRMS(sr.sector_id, sr.n_oot_cadences, sr.rms, elevated,
                                  round(z, 3) if z is not None else None))

    flag = "ALL_ELEVATED" if n_elev == len(updated) else "OK"

    return OOTRMSResult(
        n_sectors=len(updated),
        sector_rms=tuple(updated),
        median_rms=round(median, 8),
        n_elevated=n_elev,
        flag=flag,
    )


def format_oot_rms_result(result: OOTRMSResult) -> str:
    """Format OOT RMS tracking result as Markdown."""
    lines = [
        "## OOT RMS Tracker",
        "",
        f"- Sectors: {result.n_sectors}",
        f"- Median RMS: {result.median_rms}",
        f"- Elevated sectors: {result.n_elevated}",
        f"- **Flag: {result.flag}**",
    ]
    if result.sector_rms:
        lines += ["", "| Sector | N OOT | RMS | Z-score | Elevated |",
                  "|---|---|---|---|---|"]
        for s in result.sector_rms:
            lines.append(
                f"| {s.sector_id} | {s.n_oot_cadences} | {s.rms:.3e} "
                f"| {s.z_score} | {'✓' if s.is_elevated else '—'} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="oot_rms_tracker",
        description="Track per-sector OOT RMS for systematics detection.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = track_oot_rms([], [], args.period_days, args.epoch_bjd, [],
                            duration_hours=args.duration_hours)
    print(format_oot_rms_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
