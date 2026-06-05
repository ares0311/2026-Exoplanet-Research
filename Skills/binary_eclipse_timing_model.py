"""Eclipse timing model for detached EBs: O-C from light travel time (Rømer delay)."""
from __future__ import annotations

import math
from dataclasses import dataclass

_C_MS = 2.998e8
_AU_M = 1.495978707e11
_G = 6.674e-11
_MSUN_KG = 1.989e30
_SEC_PER_DAY = 86400.0


@dataclass(frozen=True)
class EclipseTimingResult:
    eclipse_number: int
    predicted_time_bjd: float
    ltt_offset_seconds: float    # light travel time O-C
    roemer_delay_seconds: float  # Rømer delay for given orbital phase
    timing_uncertainty_seconds: float
    flag: str


@dataclass(frozen=True)
class BinaryEclipseTimingModelResult:
    eclipse_times: tuple[EclipseTimingResult, ...]
    romer_semi_amplitude_seconds: float
    period_days: float
    flag: str


def compute_binary_eclipse_timing(
    epoch_bjd: float,
    period_days: float,
    n_eclipses: int = 10,
    third_body_mass_mjup: float = 0.0,
    third_body_period_yr: float = 10.0,
    third_body_eccentricity: float = 0.0,
    binary_mass_msun: float = 2.0,
) -> BinaryEclipseTimingModelResult:
    """Compute predicted eclipse times with Rømer delay from a tertiary companion.

    Rømer semi-amplitude: a₁₂ sin(i) / c
    a₁₂ = (M₃ / (M₁₂ + M₃)) × a_outer

    Args:
        epoch_bjd: reference eclipse epoch (BJD)
        period_days: eclipse period (days)
        n_eclipses: number of eclipse times to compute
        third_body_mass_mjup: third body mass (Jupiter masses); 0 = no tertiary
        third_body_period_yr: third body orbital period (years)
        third_body_eccentricity: third body eccentricity
        binary_mass_msun: total binary mass (solar masses)
    """
    _MJUP_MSUN = 1.898e27 / 1.989e30

    if period_days <= 0.0:
        return BinaryEclipseTimingModelResult((), float("nan"), period_days,
                                               "INVALID_PERIOD")
    if n_eclipses < 1:
        return BinaryEclipseTimingModelResult((), float("nan"), period_days,
                                               "INVALID_N_ECLIPSES")

    m3_msun = third_body_mass_mjup * _MJUP_MSUN
    m_total = binary_mass_msun + m3_msun

    if m3_msun > 0.0 and third_body_period_yr > 0.0:
        a_outer_au = (third_body_period_yr**2 * m_total) ** (1.0 / 3.0)
        a12_au = (m3_msun / m_total) * a_outer_au
        roemer_semi = a12_au * _AU_M / _C_MS
        p_outer_days = third_body_period_yr * 365.25
    else:
        roemer_semi = 0.0
        p_outer_days = float("inf")

    eclipse_results = []
    for i in range(n_eclipses):
        t_pred = epoch_bjd + i * period_days
        if roemer_semi > 0.0 and math.isfinite(p_outer_days):
            phase_outer = (i * period_days % p_outer_days) / p_outer_days
            eccentric_term = math.sin(2.0 * math.pi * phase_outer)
            ltt = roemer_semi * eccentric_term
        else:
            ltt = 0.0

        unc = roemer_semi * 0.1 if roemer_semi > 0.0 else 0.5

        eclipse_results.append(EclipseTimingResult(
            eclipse_number=i,
            predicted_time_bjd=t_pred,
            ltt_offset_seconds=ltt,
            roemer_delay_seconds=ltt,
            timing_uncertainty_seconds=unc,
            flag="OK",
        ))

    return BinaryEclipseTimingModelResult(
        eclipse_times=tuple(eclipse_results),
        romer_semi_amplitude_seconds=roemer_semi,
        period_days=period_days,
        flag="OK",
    )


def format_binary_eclipse_timing_result(r: BinaryEclipseTimingModelResult) -> str:
    if r.flag != "OK":
        return f"BinaryEclipseTiming | flag={r.flag}"
    lines = [
        f"Period: {r.period_days:.6f} days | "
        f"Rømer semi-amplitude: {r.romer_semi_amplitude_seconds:.1f} s | "
        f"flag={r.flag}",
        "",
        "| # | Predicted BJD | LTT O-C (s) |",
        "|---|---|---|",
    ]
    for e in r.eclipse_times[:10]:
        lines.append(
            f"| {e.eclipse_number} | {e.predicted_time_bjd:.6f} | {e.ltt_offset_seconds:.2f} |"
        )
    return "\n".join(lines)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Binary eclipse timing model (Rømer delay)")
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("period_days", type=float)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--m3-mjup", type=float, default=0.0)
    p.add_argument("--p3-yr", type=float, default=10.0)
    args = p.parse_args()
    r = compute_binary_eclipse_timing(args.epoch_bjd, args.period_days, n_eclipses=args.n,
                                       third_body_mass_mjup=args.m3_mjup,
                                       third_body_period_yr=args.p3_yr)
    print(format_binary_eclipse_timing_result(r))


if __name__ == "__main__":
    _cli()
