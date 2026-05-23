"""Compute transit ingress and egress durations from transit geometry.

Analytically derives the ingress/egress duration using the standard transit
geometry relations.  Distinct from ``transit_duration_calculator`` (total
duration only) and ``transit_geometry_calculator`` (geometric parameters
without ingress timing).

Public API
----------
IngressResult(total_duration_hours, ingress_duration_hours,
              egress_duration_hours, flat_bottom_hours,
              ingress_fraction, is_grazing, flag)
compute_ingress_duration(period_days, rp_over_rs, b, a_over_rs, *,
                         limb_darkening_u1) -> IngressResult
format_ingress_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IngressResult:
    total_duration_hours: float | None    # T14: first to last contact
    ingress_duration_hours: float | None  # T12: first to second contact
    egress_duration_hours: float | None   # T34: third to fourth contact
    flat_bottom_hours: float | None       # T23: second to third contact
    ingress_fraction: float | None        # ingress_duration / total_duration
    is_grazing: bool                      # b + Rp/Rs > 1 → no flat bottom
    flag: str  # "OK" | "INVALID"


def compute_ingress_duration(
    period_days: float,
    rp_over_rs: float,
    b: float,
    a_over_rs: float,
    *,
    limb_darkening_u1: float = 0.4,
) -> IngressResult:
    """Compute ingress/egress duration from transit geometry.

    Uses the Seager & Mallén-Ornelas (2003) analytic formulae:
    - T14 = (P/π) × arcsin((1/a_over_rs) × sqrt((1+k)² - b²) / sin(i))
    - T23 = (P/π) × arcsin((1/a_over_rs) × sqrt((1-k)² - b²) / sin(i))
    - τ = (T14 - T23) / 2  (ingress = egress for circular orbit)

    Args:
        period_days: Orbital period (days).
        rp_over_rs: Planet-to-star radius ratio Rp/Rs.
        b: Impact parameter (0 = central transit, 1 = grazing).
        a_over_rs: Scaled semi-major axis a/Rs.
        limb_darkening_u1: Linear LD coefficient (not used in duration but
            stored for context; affects shape not duration at this level).

    Returns:
        :class:`IngressResult`.
    """
    if period_days <= 0 or rp_over_rs < 0 or a_over_rs <= 0:
        return IngressResult(None, None, None, None, None, False, "INVALID")
    if b < 0 or b > 1 + rp_over_rs:
        return IngressResult(None, None, None, None, None, False, "INVALID")

    k = rp_over_rs
    # Inclination from impact parameter: cos(i) = b / a_over_rs
    cos_i = b / a_over_rs
    cos_i = max(-1.0, min(1.0, cos_i))
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i ** 2))
    if sin_i < 1e-10:
        return IngressResult(None, None, None, None, None, True, "INVALID")

    is_grazing = (b + k) >= 1.0

    # T14: first to last contact
    arg14 = ((1.0 + k) ** 2 - b ** 2)
    if arg14 < 0:
        return IngressResult(None, None, None, None, None, is_grazing, "INVALID")
    t14_days = (period_days / math.pi) * math.asin(
        math.sqrt(arg14) / (a_over_rs * sin_i)
    )

    # T23: second to third contact (flat bottom)
    arg23 = ((1.0 - k) ** 2 - b ** 2)
    t23_days: float | None = None
    if arg23 >= 0 and not is_grazing:
        t23_days = (period_days / math.pi) * math.asin(
            math.sqrt(arg23) / (a_over_rs * sin_i)
        )

    t14_hours = t14_days * 24.0
    t23_hours = t23_days * 24.0 if t23_days is not None else None

    if t23_hours is not None:
        ingress_hours = (t14_hours - t23_hours) / 2.0
        egress_hours = ingress_hours
        flat_bottom_hours = t23_hours
    else:
        ingress_hours = t14_hours / 4.0  # rough estimate for grazing
        egress_hours = ingress_hours
        flat_bottom_hours = 0.0

    ingress_frac = round(ingress_hours / t14_hours, 6) if t14_hours > 0 else None

    return IngressResult(
        total_duration_hours=round(t14_hours, 6),
        ingress_duration_hours=round(ingress_hours, 6),
        egress_duration_hours=round(egress_hours, 6),
        flat_bottom_hours=round(flat_bottom_hours, 6) if flat_bottom_hours is not None else None,
        ingress_fraction=ingress_frac,
        is_grazing=is_grazing,
        flag="OK",
    )


def format_ingress_result(result: IngressResult) -> str:
    """Format ingress duration result as Markdown."""
    lines = [
        "## Transit Ingress Timer",
        "",
        f"- Total duration T14: {result.total_duration_hours} hours",
        f"- **Ingress duration τ: {result.ingress_duration_hours} hours**",
        f"- Egress duration: {result.egress_duration_hours} hours",
        f"- Flat bottom T23: {result.flat_bottom_hours} hours",
        f"- Ingress fraction: {result.ingress_fraction}",
        f"- Grazing transit: {'Yes' if result.is_grazing else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_ingress_timer",
        description="Compute transit ingress/egress duration from geometry.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("rp_over_rs", type=float)
    parser.add_argument("b", type=float)
    parser.add_argument("a_over_rs", type=float)
    args = parser.parse_args(argv)

    result = compute_ingress_duration(args.period_days, args.rp_over_rs, args.b, args.a_over_rs)
    print(format_ingress_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
