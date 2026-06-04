"""Normalize and combine light curves from multiple instruments."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CombinedLcResult:
    n_instruments: int
    n_points_total: int
    offsets: tuple[float, ...]        # additive offset applied per instrument
    rms_per_instrument: tuple[float, ...]
    combined_rms: float
    flag: str


def combine_lightcurves(
    flux_arrays: list[list[float]],
    instrument_ids: list[str] | None = None,
    normalize: bool = True,
) -> CombinedLcResult:
    """
    Normalize and offset-correct multi-instrument flux arrays.

    Each instrument's flux is normalized to zero median (or unit median if
    normalize=False is used for absolute flux). An additive offset is applied
    so all instruments share the same baseline.

    Parameters
    ----------
    flux_arrays:     List of flux arrays, one per instrument.
    instrument_ids:  Optional names for each instrument.
    normalize:       If True, subtract median from each array (differential).
    """
    n = len(flux_arrays)
    if n < 1:
        return CombinedLcResult(
            n_instruments=0, n_points_total=0, offsets=(),
            rms_per_instrument=(), combined_rms=float("nan"), flag="NO_INSTRUMENTS",
        )

    if instrument_ids is None:
        instrument_ids = [f"inst_{i}" for i in range(n)]

    offsets: list[float] = []
    rms_list: list[float] = []
    combined: list[float] = []

    for arr in flux_arrays:
        finite = [f for f in arr if math.isfinite(f)]
        if not finite:
            offsets.append(0.0)
            rms_list.append(float("nan"))
            continue

        sorted_f = sorted(finite)
        mid = len(sorted_f) // 2
        median = (
            sorted_f[mid] if len(sorted_f) % 2
            else (sorted_f[mid - 1] + sorted_f[mid]) / 2.0
        )
        offset = median if normalize else 0.0
        offsets.append(round(offset, 6))

        detrended = [f - offset for f in finite]
        rms = math.sqrt(sum(d ** 2 for d in detrended) / len(detrended))
        rms_list.append(round(rms, 6))
        combined.extend(detrended)

    c_rms = math.sqrt(sum(d ** 2 for d in combined) / len(combined)) if combined else float("nan")

    return CombinedLcResult(
        n_instruments=n,
        n_points_total=len(combined),
        offsets=tuple(offsets),
        rms_per_instrument=tuple(rms_list),
        combined_rms=round(c_rms, 6) if math.isfinite(c_rms) else float("nan"),
        flag="OK",
    )


def format_combined_lc_result(r: CombinedLcResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.6f}" if math.isfinite(v) else "N/A"

    lines = [
        "| Parameter | Value |\n|---|---|",
        f"| N instruments | {r.n_instruments} |",
        f"| N points total | {r.n_points_total} |",
        f"| Combined RMS | {_f(r.combined_rms)} |",
        f"| Flag | {r.flag} |",
        "",
        "| Instrument | Offset | RMS |",
        "|---|---|---|",
    ]
    for i, (off, rms) in enumerate(zip(r.offsets, r.rms_per_instrument, strict=True)):
        lines.append(f"| inst_{i} | {_f(off)} | {_f(rms)} |")
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Combine multi-instrument light curves.")
    p.add_argument("fluxes_json", help="JSON array of arrays (one per instrument)")
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()
    import json
    arrays = json.loads(args.fluxes_json)
    r = combine_lightcurves(arrays, normalize=not args.no_normalize)
    print(format_combined_lc_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
