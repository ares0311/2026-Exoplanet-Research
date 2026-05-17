"""Bin a light curve to a coarser cadence and export as JSON.

Useful for visualisation, compressed storage, and feeding slower algorithms.

Public API
----------
BinnedLC(time, flux, flux_err, n_points_per_bin, cadence_minutes)
bin_lightcurve(time, flux, *, flux_err, bin_minutes) -> BinnedLC
export_binned_lc(binned, path) -> Path
load_binned_lc(path) -> BinnedLC
format_bin_summary(binned) -> str
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BinnedLC:
    time: tuple[float, ...]
    flux: tuple[float, ...]
    flux_err: tuple[float, ...]
    n_points_per_bin: tuple[int, ...]
    cadence_minutes: float


def bin_lightcurve(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    bin_minutes: float = 30.0,
) -> BinnedLC:
    """Bin time-series data into fixed-width time bins.

    Args:
        time: BJD time array (must be sorted ascending).
        flux: Flux array.
        flux_err: Per-cadence uncertainties; if None, uses scatter within bin.
        bin_minutes: Target bin width in minutes.

    Returns:
        :class:`BinnedLC` with weighted-mean flux per bin.
    """
    if not time:
        return BinnedLC((), (), (), (), bin_minutes)

    t0 = float(time[0])
    bin_days = bin_minutes / 1440.0

    bins: dict[int, list] = {}
    for i, (t, f) in enumerate(zip(time, flux, strict=False)):
        bin_idx = int((float(t) - t0) / bin_days)
        e = float(flux_err[i]) if flux_err is not None else None
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append((float(t), float(f), e))

    out_t: list[float] = []
    out_f: list[float] = []
    out_e: list[float] = []
    out_n: list[int] = []

    for idx in sorted(bins):
        points = bins[idx]
        n = len(points)
        if all(p[2] is not None for p in points):
            weights = [1.0 / max(p[2] ** 2, 1e-30) for p in points]
            w_sum = sum(weights)
            t_mean = sum(p[0] * w for p, w in zip(points, weights, strict=False)) / w_sum
            f_mean = sum(p[1] * w for p, w in zip(points, weights, strict=False)) / w_sum
            f_err = 1.0 / math.sqrt(w_sum)
        else:
            t_mean = sum(p[0] for p in points) / n
            f_mean = sum(p[1] for p in points) / n
            sq = sum((p[1] - f_mean) ** 2 for p in points)
            f_err = math.sqrt(sq / n) / math.sqrt(n) if n > 1 else 0.0

        out_t.append(t_mean)
        out_f.append(f_mean)
        out_e.append(f_err)
        out_n.append(n)

    return BinnedLC(
        time=tuple(out_t),
        flux=tuple(out_f),
        flux_err=tuple(out_e),
        n_points_per_bin=tuple(out_n),
        cadence_minutes=bin_minutes,
    )


def export_binned_lc(binned: BinnedLC, path: Path | str) -> Path:
    """Write binned LC to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "time": list(binned.time),
        "flux": list(binned.flux),
        "flux_err": list(binned.flux_err),
        "n_points_per_bin": list(binned.n_points_per_bin),
        "cadence_minutes": binned.cadence_minutes,
    }
    p.write_text(json.dumps(data, indent=2))
    return p


def load_binned_lc(path: Path | str) -> BinnedLC:
    """Load a BinnedLC from a JSON file."""
    data = json.loads(Path(path).read_text())
    return BinnedLC(
        time=tuple(data["time"]),
        flux=tuple(data["flux"]),
        flux_err=tuple(data["flux_err"]),
        n_points_per_bin=tuple(data["n_points_per_bin"]),
        cadence_minutes=float(data["cadence_minutes"]),
    )


def format_bin_summary(binned: BinnedLC) -> str:
    """Format summary of a binned LC as Markdown."""
    n = len(binned.time)
    mean_n = (sum(binned.n_points_per_bin) / n) if n > 0 else 0.0
    lines = [
        "## Binned Light Curve Summary",
        "",
        f"- Bins: {n}",
        f"- Bin width: {binned.cadence_minutes:.1f} min",
        f"- Mean points per bin: {mean_n:.1f}",
    ]
    if binned.time:
        lines.append(f"- Time span: {binned.time[0]:.2f} – {binned.time[-1]:.2f} BJD")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="binned_lc_exporter",
        description="Bin a light curve to coarser cadence and export JSON.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--bin-minutes", type=float, default=30.0)
    parser.add_argument("--output", required=True, metavar="JSON")
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    binned = bin_lightcurve(lc["time"], lc["flux"], bin_minutes=args.bin_minutes)
    export_binned_lc(binned, args.output)
    print(format_bin_summary(binned))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
