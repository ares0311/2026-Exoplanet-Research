"""Update transit ephemeris from new transit midpoint measurements.

Fits a linear O-C model to observed minus computed transit times to
refine the period and epoch.

Public API
----------
EphemerisUpdate(original_period, original_epoch, fitted_period,
                fitted_epoch, period_correction, epoch_correction,
                rms_oc_min, n_transits, flag)
update_ephemeris(midpoints, *, period_days, epoch_btjd) -> EphemerisUpdate
format_ephemeris_update(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EphemerisUpdate:
    original_period: float
    original_epoch: float
    fitted_period: float
    fitted_epoch: float
    period_correction: float   # days; fitted - original
    epoch_correction: float    # days; fitted - original
    rms_oc_min: float          # O-C residual RMS in minutes
    n_transits: int
    flag: str  # "OK" | "IMPROVED" | "SPARSE" | "INVALID"


def update_ephemeris(
    midpoints: list[float] | tuple[float, ...],
    *,
    period_days: float,
    epoch_btjd: float,
) -> EphemerisUpdate:
    """Fit a linear O-C model to refine period and epoch.

    Uses least-squares fit of transit number vs. observed midpoint to
    compute corrected period dP and corrected epoch dT0.

    Args:
        midpoints: Observed transit midpoint times in BTJD.
        period_days: Current best-fit orbital period.
        epoch_btjd: Current best-fit epoch.

    Returns:
        EphemerisUpdate with fitted period and epoch.
    """
    mpts = list(midpoints)
    n = len(mpts)

    if period_days <= 0:
        return EphemerisUpdate(
            original_period=period_days,
            original_epoch=epoch_btjd,
            fitted_period=period_days,
            fitted_epoch=epoch_btjd,
            period_correction=0.0,
            epoch_correction=0.0,
            rms_oc_min=0.0,
            n_transits=n,
            flag="INVALID",
        )

    if n < 2:
        return EphemerisUpdate(
            original_period=period_days,
            original_epoch=epoch_btjd,
            fitted_period=period_days,
            fitted_epoch=epoch_btjd,
            period_correction=0.0,
            epoch_correction=0.0,
            rms_oc_min=0.0,
            n_transits=n,
            flag="SPARSE",
        )

    # Compute integer transit numbers
    transit_ns = [
        round((t - epoch_btjd) / period_days) for t in mpts
    ]

    # O-C residuals with original ephemeris
    oc = [(mpts[i] - (epoch_btjd + transit_ns[i] * period_days)) * 1440.0
          for i in range(n)]

    # Weighted least-squares: OC_i = dT0 + dP * E_i
    # Normal equations
    sum_e = sum(float(e) for e in transit_ns)
    sum_e2 = sum(float(e) ** 2 for e in transit_ns)
    sum_oc = sum(oc)
    sum_eoc = sum(float(transit_ns[i]) * oc[i] for i in range(n))
    det = n * sum_e2 - sum_e ** 2

    if abs(det) < 1e-12:
        return EphemerisUpdate(
            original_period=period_days,
            original_epoch=epoch_btjd,
            fitted_period=period_days,
            fitted_epoch=epoch_btjd,
            period_correction=0.0,
            epoch_correction=0.0,
            rms_oc_min=0.0,
            n_transits=n,
            flag="INVALID",
        )

    dt0_min = (sum_e2 * sum_oc - sum_e * sum_eoc) / det
    dp_min = (n * sum_eoc - sum_e * sum_oc) / det

    dt0_days = dt0_min / 1440.0
    dp_days = dp_min / 1440.0

    fitted_epoch = epoch_btjd + dt0_days
    fitted_period = period_days + dp_days

    # Post-fit residuals
    residuals = [
        (mpts[i] - (fitted_epoch + transit_ns[i] * fitted_period)) * 1440.0
        for i in range(n)
    ]
    rms = math.sqrt(sum(r ** 2 for r in residuals) / n)

    flag = "IMPROVED" if abs(dp_days) > 1e-7 or abs(dt0_days) > 1e-6 else "OK"

    return EphemerisUpdate(
        original_period=period_days,
        original_epoch=epoch_btjd,
        fitted_period=round(fitted_period, 8),
        fitted_epoch=round(fitted_epoch, 6),
        period_correction=round(dp_days, 8),
        epoch_correction=round(dt0_days, 6),
        rms_oc_min=round(rms, 4),
        n_transits=n,
        flag=flag,
    )


def format_ephemeris_update(result: EphemerisUpdate) -> str:
    """Format ephemeris update as Markdown.

    Args:
        result: EphemerisUpdate to format.

    Returns:
        Markdown string.
    """
    dp_str = f"{result.period_correction * 86400:.2f} s"
    dt0_str = f"{result.epoch_correction * 1440:.2f} min"
    lines = [
        "## Transit Ephemeris Update\n",
        f"**Status**: `{result.flag}` | N transits: {result.n_transits} | "
        f"O-C RMS: {result.rms_oc_min:.2f} min\n",
        "",
        "| Parameter | Original | Updated | Correction |",
        "|---|---|---|---|",
        f"| Period (d) | {result.original_period:.8f} | "
        f"{result.fitted_period:.8f} | {dp_str} |",
        f"| Epoch (BTJD) | {result.original_epoch:.6f} | "
        f"{result.fitted_epoch:.6f} | {dt0_str} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Update transit ephemeris.")
    parser.add_argument("midpoints", help="JSON file with list of midpoint times.")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.midpoints).read_text())
    result = update_ephemeris(data, period_days=args.period, epoch_btjd=args.epoch)
    print(format_ephemeris_update(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
