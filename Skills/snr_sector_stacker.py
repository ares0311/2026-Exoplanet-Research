"""Predict combined detection SNR from stacking multiple TESS sectors.

Analytic forecasting tool — no light curve data required.  Uses the
sqrt(N) scaling law with optional per-sector noise variation to project
how the transit detection SNR will improve as more sectors are added.

Public API
----------
StackedSNRResult(snr_single, n_sectors, snr_stacked, snr_gain,
                 sectors_for_threshold, flag)
project_stacked_snr(snr_single_sector, n_sectors, *,
                    per_sector_noise, snr_threshold) -> StackedSNRResult
format_stacked_snr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StackedSNRResult:
    snr_single: float           # SNR in a single sector
    n_sectors: int              # number of sectors to stack
    snr_stacked: float          # projected combined SNR
    snr_gain: float             # snr_stacked / snr_single
    sectors_for_threshold: int | None  # N to reach snr_threshold (None if already reached)
    flag: str  # "OK" | "INVALID"


def project_stacked_snr(
    snr_single_sector: float,
    n_sectors: int,
    *,
    per_sector_noise: list[float] | None = None,
    snr_threshold: float = 7.1,
) -> StackedSNRResult:
    """Project combined SNR from stacking N TESS sectors.

    When ``per_sector_noise`` is provided, each sector's contribution is
    weighted by its inverse noise variance (heterogeneous stacking).
    Otherwise assumes uniform noise across all sectors (sqrt(N) scaling).

    Args:
        snr_single_sector: Measured or estimated SNR in a single reference sector.
        n_sectors: Number of sectors to stack.
        per_sector_noise: Optional list of per-sector relative noise values
            (length must equal n_sectors; values normalised internally).
        snr_threshold: SNR threshold for flagging how many sectors are needed.

    Returns:
        :class:`StackedSNRResult`.
    """
    if snr_single_sector <= 0 or n_sectors < 1:
        return StackedSNRResult(snr_single_sector, n_sectors, 0.0, 0.0, None, "INVALID")

    if per_sector_noise and len(per_sector_noise) == n_sectors:
        # Heterogeneous: SNR_stack = SNR_1 * sqrt(sum(w_i)) / sqrt(w_1)
        # where w_i = 1 / sigma_i^2, normalised so w_1 = 1
        noise_ref = per_sector_noise[0] if per_sector_noise[0] > 0 else 1.0
        w_ref = 1.0 / (noise_ref ** 2)
        w_sum = sum(1.0 / (s ** 2) if s > 0 else 0.0 for s in per_sector_noise)
        gain = math.sqrt(w_sum / w_ref) if w_ref > 0 else math.sqrt(n_sectors)
    else:
        gain = math.sqrt(n_sectors)

    snr_stacked = snr_single_sector * gain

    # How many sectors to reach threshold?
    if snr_single_sector >= snr_threshold:
        sectors_needed: int | None = 1
    else:
        n_needed = math.ceil((snr_threshold / snr_single_sector) ** 2)
        sectors_needed = int(n_needed)

    return StackedSNRResult(
        snr_single=round(snr_single_sector, 4),
        n_sectors=n_sectors,
        snr_stacked=round(snr_stacked, 4),
        snr_gain=round(gain, 4),
        sectors_for_threshold=sectors_needed,
        flag="OK",
    )


def format_stacked_snr_result(result: StackedSNRResult) -> str:
    """Format stacked SNR projection as Markdown."""
    lines = [
        "## Sector SNR Stacker",
        "",
        f"- Single-sector SNR: {result.snr_single}",
        f"- Sectors stacked: {result.n_sectors}",
        f"- **Projected stacked SNR: {result.snr_stacked:.2f}**",
        f"- SNR gain: {result.snr_gain:.3f}×",
    ]
    if result.sectors_for_threshold is not None:
        lines.append(f"- Sectors needed to reach threshold: {result.sectors_for_threshold}")
    lines.append(f"- **Flag: {result.flag}**")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="snr_sector_stacker",
        description="Project combined SNR from stacking TESS sectors.",
    )
    parser.add_argument("snr_single", type=float)
    parser.add_argument("n_sectors", type=int)
    parser.add_argument("--snr-threshold", type=float, default=7.1)
    args = parser.parse_args(argv)

    result = project_stacked_snr(
        args.snr_single, args.n_sectors, snr_threshold=args.snr_threshold
    )
    print(format_stacked_snr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
