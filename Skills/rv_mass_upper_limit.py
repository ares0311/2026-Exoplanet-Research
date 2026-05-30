from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvMassLimitResult:
    mass_upper_limit_mjup: float
    k_amplitude_ms: float
    period_days: float
    flag: str


def compute_rv_upper_limit(
    rv_precision_ms: float,
    period_days: float,
    stellar_mass_msun: float,
    *,
    n_sigma: float = 3.0,
    inclination_deg: float = 90.0,
) -> RvMassLimitResult:
    """Compute an upper limit on planet mass from RV precision.

    Uses the RV semi-amplitude relation:
        K [m/s] ≈ 203.3 * (M_p/M_jup) * sin(i) / (P[days]^(1/3) * (M★/M☉)^(2/3))

    Rearranged:
        M_p = K * P^(1/3) * M★^(2/3) / (203.3 * sin(i))
    """
    if rv_precision_ms <= 0:
        return RvMassLimitResult(
            mass_upper_limit_mjup=0.0,
            k_amplitude_ms=0.0,
            period_days=period_days,
            flag="INVALID_PRECISION",
        )

    if period_days <= 0:
        return RvMassLimitResult(
            mass_upper_limit_mjup=0.0,
            k_amplitude_ms=0.0,
            period_days=period_days,
            flag="INVALID_PERIOD",
        )

    if stellar_mass_msun <= 0:
        return RvMassLimitResult(
            mass_upper_limit_mjup=0.0,
            k_amplitude_ms=0.0,
            period_days=period_days,
            flag="INVALID_STELLAR_MASS",
        )

    k_amplitude_ms = rv_precision_ms * n_sigma
    sin_i = math.sin(inclination_deg * math.pi / 180.0)

    mass_upper_limit_mjup = (
        k_amplitude_ms
        * (period_days ** (1.0 / 3.0))
        * (stellar_mass_msun ** (2.0 / 3.0))
        / (203.3 * sin_i)
    )

    return RvMassLimitResult(
        mass_upper_limit_mjup=mass_upper_limit_mjup,
        k_amplitude_ms=k_amplitude_ms,
        period_days=period_days,
        flag="OK",
    )


def format_rv_mass_limit(result: RvMassLimitResult) -> str:
    """Return a Markdown table summarising the RV mass upper limit result."""
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Mass Upper Limit (M_Jup) | {result.mass_upper_limit_mjup:.4f} |",
        f"| K Amplitude (m/s) | {result.k_amplitude_ms:.4f} |",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute RV-based planet mass upper limit."
    )
    parser.add_argument(
        "rv_precision_ms", type=float, help="RV measurement precision in m/s."
    )
    parser.add_argument("period_days", type=float, help="Orbital period in days.")
    parser.add_argument(
        "stellar_mass_msun", type=float, help="Stellar mass in solar masses."
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=3.0,
        help="Detection threshold in sigma (default 3.0).",
    )
    parser.add_argument(
        "--inclination-deg",
        type=float,
        default=90.0,
        help="Orbital inclination in degrees (default 90.0).",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    result = compute_rv_upper_limit(
        args.rv_precision_ms,
        args.period_days,
        args.stellar_mass_msun,
        n_sigma=args.n_sigma,
        inclination_deg=args.inclination_deg,
    )

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_rv_mass_limit(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
