"""Per-sector transit depth SNR for prioritising vetting effort.

For each TESS sector computes the weighted-mean in-transit depth and its
signal-to-noise ratio relative to the out-of-transit scatter.  Sectors with
low SNR are unlikely to contribute useful vetting information; sectors with
anomalously different depths flag potential systematics.

Public API
----------
SectorDepthSNR(sector_id, n_transits, depth_ppm, depth_err_ppm, snr,
               is_reliable)
DepthSNRResult(n_sectors, depths_ppm, mean_depth_ppm, depth_rms_ppm,
               sector_snrs, flag)
compute_depth_snr_per_sector(time, flux, period_days, epoch_bjd, sector_ids,
                              *, flux_err, duration_hours,
                              min_snr) -> DepthSNRResult
format_depth_snr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorDepthSNR:
    sector_id: int
    n_in_transit: int
    depth_ppm: float
    depth_err_ppm: float | None
    snr: float | None
    is_reliable: bool     # SNR >= min_snr


@dataclass(frozen=True)
class DepthSNRResult:
    n_sectors: int
    sector_snrs: tuple[SectorDepthSNR, ...]
    mean_depth_ppm: float | None
    depth_rms_ppm: float | None   # scatter of per-sector depths
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def compute_depth_snr_per_sector(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    sector_ids: list[int],
    *,
    flux_err: list[float] | None = None,
    duration_hours: float = 2.0,
    min_snr: float = 3.0,
) -> DepthSNRResult:
    """Compute per-sector depth and SNR.

    Args:
        time: Time array (BJD or BTJD).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch.
        sector_ids: Integer sector label per cadence.
        flux_err: Per-point uncertainties (optional).
        duration_hours: Transit duration in hours.
        min_snr: SNR threshold for ``is_reliable``.

    Returns:
        :class:`DepthSNRResult`.
    """
    n = len(flux)
    if n < 5 or len(time) != n or len(sector_ids) != n or period_days <= 0:
        return DepthSNRResult(0, (), None, None, "INVALID")

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n
    half_width = (duration_hours / 24.0) / period_days / 2.0
    oot_half = min(3 * half_width, 0.4)

    unique = sorted(set(sector_ids))
    phases = _phase_fold(time, epoch_bjd, period_days)

    sector_snrs: list[SectorDepthSNR] = []
    depths: list[float] = []

    for sec in unique:
        idx = [i for i in range(n) if sector_ids[i] == sec]
        if not idx:
            continue

        in_f: list[float] = []
        in_w: list[float] = []
        oot_f: list[float] = []
        oot_w: list[float] = []

        for i in idx:
            ph = phases[i]
            ap = abs(ph)
            w = 1.0 / max(errs[i] ** 2, 1e-30)
            if ap <= half_width:
                in_f.append(flux[i])
                in_w.append(w)
            elif half_width < ap <= oot_half:
                oot_f.append(flux[i])
                oot_w.append(w)

        if len(in_f) < 2 or len(oot_f) < 2:
            sector_snrs.append(SectorDepthSNR(sec, len(in_f), 0.0, None, None, False))
            continue

        def _wmean(vals: list[float], ws: list[float]) -> float:
            sw = sum(ws)
            return sum(v * w for v, w in zip(vals, ws, strict=False)) / sw if sw > 0 else 0.0

        baseline = _wmean(oot_f, oot_w)
        in_mean = _wmean(in_f, in_w)
        depth = (baseline - in_mean) * 1e6  # ppm

        # Depth uncertainty: propagate error from in_mean variance
        sum_w_in = sum(in_w)
        depth_err_ppm = (1.0 / math.sqrt(max(sum_w_in, 1e-30))) * 1e6 if sum_w_in > 0 else None

        snr = abs(depth) / depth_err_ppm if depth_err_ppm and depth_err_ppm > 0 else None

        sector_snrs.append(SectorDepthSNR(
            sector_id=sec,
            n_in_transit=len(in_f),
            depth_ppm=round(depth, 2),
            depth_err_ppm=round(depth_err_ppm, 2) if depth_err_ppm is not None else None,
            snr=round(snr, 2) if snr is not None else None,
            is_reliable=snr is not None and snr >= min_snr,
        ))
        if depth > 0:
            depths.append(depth)

    if not sector_snrs:
        return DepthSNRResult(0, (), None, None, "INSUFFICIENT")

    mean_d = sum(depths) / len(depths) if depths else None
    rms_d: float | None = None
    if len(depths) > 1:
        mean_d2 = sum(depths) / len(depths)
        rms_d = math.sqrt(sum((d - mean_d2) ** 2 for d in depths) / len(depths))

    return DepthSNRResult(
        n_sectors=len(sector_snrs),
        sector_snrs=tuple(sector_snrs),
        mean_depth_ppm=round(mean_d, 2) if mean_d is not None else None,
        depth_rms_ppm=round(rms_d, 2) if rms_d is not None else None,
        flag="OK",
    )


def format_depth_snr_result(result: DepthSNRResult) -> str:
    """Format per-sector depth SNR as Markdown."""
    lines = [
        "## Per-Sector Depth SNR",
        "",
        f"- Sectors: {result.n_sectors}",
        f"- Mean depth: {result.mean_depth_ppm} ppm",
        f"- Depth RMS: {result.depth_rms_ppm} ppm",
        f"- **Flag: {result.flag}**",
        "",
        "| Sector | N in-transit | Depth (ppm) | Depth err | SNR | Reliable |",
        "|---|---|---|---|---|---|",
    ]
    for s in result.sector_snrs:
        lines.append(
            f"| {s.sector_id} | {s.n_in_transit} | {s.depth_ppm:.1f} "
            f"| {s.depth_err_ppm} | {s.snr} | {'✓' if s.is_reliable else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="depth_snr_per_sector",
        description="Compute per-sector transit depth SNR.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--min-snr", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = compute_depth_snr_per_sector(
        [], [], args.period_days, args.epoch_bjd, [],
        duration_hours=args.duration_hours,
        min_snr=args.min_snr,
    )
    print(format_depth_snr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
