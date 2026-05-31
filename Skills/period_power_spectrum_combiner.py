"""Combine BLS power spectra from multiple sectors."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CombinedSpectrumResult:
    n_spectra: int
    n_periods: int
    peak_period: float
    peak_power: float
    peak_index: int
    flag: str


def combine_power_spectra(
    periods: list[float],
    power_arrays: list[list[float]],
    *,
    normalize: bool = True,
) -> CombinedSpectrumResult:
    """
    Normalize each sector's BLS power array (0–1) then sum; find the combined peak.
    """
    n = len(periods)
    if n < 2:
        return CombinedSpectrumResult(
            n_spectra=len(power_arrays), n_periods=n,
            peak_period=float("nan"), peak_power=float("nan"),
            peak_index=-1, flag="INSUFFICIENT_PERIODS",
        )
    if not power_arrays:
        return CombinedSpectrumResult(
            n_spectra=0, n_periods=n,
            peak_period=float("nan"), peak_power=float("nan"),
            peak_index=-1, flag="NO_SPECTRA",
        )
    for arr in power_arrays:
        if len(arr) != n:
            return CombinedSpectrumResult(
                n_spectra=len(power_arrays), n_periods=n,
                peak_period=float("nan"), peak_power=float("nan"),
                peak_index=-1, flag="LENGTH_MISMATCH",
            )

    combined = [0.0] * n
    for arr in power_arrays:
        if normalize:
            pmax = max(arr)
            pmin = min(arr)
            span = pmax - pmin
            normed = [(v - pmin) / span for v in arr] if span > 0.0 else [0.0] * n
        else:
            normed = list(arr)
        for i in range(n):
            combined[i] += normed[i]

    peak_idx = max(range(n), key=lambda i: combined[i])
    return CombinedSpectrumResult(
        n_spectra=len(power_arrays),
        n_periods=n,
        peak_period=round(periods[peak_idx], 6),
        peak_power=round(combined[peak_idx], 6),
        peak_index=peak_idx,
        flag="OK",
    )


def format_combined_spectrum_result(r: CombinedSpectrumResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Spectra combined | {r.n_spectra} |\n"
        f"| Period grid points | {r.n_periods} |\n"
        f"| Peak period (days) | {r.peak_period:.6f} |\n"
        f"| Peak combined power | {r.peak_power:.6f} |\n"
        f"| Peak index | {r.peak_index} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Combine BLS power spectra from multiple sectors.")
    p.add_argument("input", help="JSON file: {periods: [...], power_arrays: [[...], ...]}")
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    r = combine_power_spectra(
        data["periods"], data["power_arrays"], normalize=not args.no_normalize
    )
    print(format_combined_spectrum_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
