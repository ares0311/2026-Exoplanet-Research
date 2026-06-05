"""Estimate stellar rotation period from age and colour via gyrochronology."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GyrochronologyResult:
    rotation_period_days: float     # predicted P_rot
    rossby_number: float            # Ro = P_rot / τ_conv
    activity_level: str             # ACTIVE / MODERATE / INACTIVE / SATURATED
    age_from_period_gyr: float | None  # inferred age if period given
    flag: str


def compute_rotation_period(
    stellar_age_gyr: float,
    bv_color: float,
    observed_period_days: float | None = None,
) -> GyrochronologyResult:
    """Estimate P_rot from stellar age and B-V colour (Barnes 2010 calibration).

    Barnes (2010) gyrochronology:
      P_rot = t^n × (a × (B-V - c)^b + f)
    with n=0.5189, a=0.7725, b=0.601, c=0.495, f=0.601
    (calibrated on open clusters up to solar age)

    Convective turnover time (Noyes+1984 approximation):
      log(τ_conv) = 1.362 - 0.166x + 0.025x² - 5.323x³ + ... (x = 1 - B-V)

    Args:
        stellar_age_gyr: stellar age (Gyr)
        bv_color: B-V colour index
        observed_period_days: if given, invert for age estimate
    """
    if stellar_age_gyr <= 0.0:
        return GyrochronologyResult(float("nan"), float("nan"), "UNKNOWN", None,
                                     "INVALID_AGE")
    if bv_color <= 0.0:
        return GyrochronologyResult(float("nan"), float("nan"), "UNKNOWN", None,
                                     "INVALID_COLOR")

    # Barnes (2010) coefficients
    n = 0.5189
    a = 0.7725
    b = 0.601
    c = 0.495
    f = 0.601

    age_myr = stellar_age_gyr * 1000.0
    bv_eff = max(bv_color - c, 0.001)

    p_rot = (age_myr ** n) * (a * bv_eff ** b + f)

    # Convective turnover time (Noyes+1984 polynomial)
    x = 1.0 - bv_color
    # Valid range: 0.5 <= B-V <= 1.4 (FGK stars)
    log_tau = (1.362 - 0.166 * x + 0.025 * x**2 - 5.323 * x**3
               if bv_color >= 0.5
               else 1.362 - 0.14 * x)
    tau_conv = 10.0 ** log_tau  # days

    rossby = p_rot / tau_conv if tau_conv > 0 else float("nan")

    # Activity level from Rossby number
    if rossby < 0.1:
        activity = "SATURATED"
    elif rossby < 0.3:
        activity = "ACTIVE"
    elif rossby < 1.0:
        activity = "MODERATE"
    else:
        activity = "INACTIVE"

    # If observed period is provided, use it as rotation_period_days and derive age
    age_inferred = None
    if observed_period_days is not None and observed_period_days > 0:
        p_rot = observed_period_days  # use observed period directly
        rossby = p_rot / tau_conv if tau_conv > 0 else float("nan")
        # Recompute activity with observed Rossby
        if rossby < 0.1:
            activity = "SATURATED"
        elif rossby < 0.3:
            activity = "ACTIVE"
        elif rossby < 1.0:
            activity = "MODERATE"
        else:
            activity = "INACTIVE"
        # Invert Barnes relation for age
        p_i_component = a * bv_eff ** b + f
        if p_i_component > 0:
            age_myr_inferred = (p_rot / p_i_component) ** (1.0 / n)
            age_inferred = age_myr_inferred / 1000.0

    return GyrochronologyResult(
        rotation_period_days=p_rot,
        rossby_number=rossby,
        activity_level=activity,
        age_from_period_gyr=age_inferred,
        flag="OK",
    )


def format_gyrochronology_result(r: GyrochronologyResult) -> str:
    if r.flag != "OK":
        return f"Gyrochronology | flag={r.flag}"
    age_str = f"{r.age_from_period_gyr:.2f} Gyr" if r.age_from_period_gyr is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| P_rot | {r.rotation_period_days:.2f} d |\n"
        f"| Rossby number | {r.rossby_number:.3f} |\n"
        f"| Activity level | {r.activity_level} |\n"
        f"| Age from P_rot | {age_str} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Gyrochronology rotation period estimator")
    p.add_argument("age_gyr", type=float)
    p.add_argument("bv_color", type=float)
    p.add_argument("--obs-period", type=float, default=None)
    args = p.parse_args()
    r = compute_rotation_period(args.age_gyr, args.bv_color,
                                 observed_period_days=args.obs_period)
    print(format_gyrochronology_result(r))


if __name__ == "__main__":
    _cli()
