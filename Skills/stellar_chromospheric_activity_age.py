"""Estimate stellar age and RV jitter from chromospheric activity log R'HK."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChromosphericActivityResult:
    log_rhk: float
    age_gyr: float
    age_uncertainty_gyr: float
    rv_jitter_ms: float
    activity_level: str   # VERY_ACTIVE / ACTIVE / MODERATE / QUIET / VERY_QUIET
    flag: str


def compute_chromospheric_age(
    log_rhk: float,
    bv_color: float = 0.65,
) -> ChromosphericActivityResult:
    """Estimate stellar age from log R'HK using Mamajek & Hillenbrand (2008).

    Age relation (M&H 2008, Eq 3):
      log(age) = -38.053 - 17.912*log(R'HK) - 1.6675*(log(R'HK))^2  [solar-type]

    RV jitter estimate from Isaacson & Fischer (2010):
      σ_jitter ≈ 10^(0.51*log R'HK + 4.72) m/s

    Args:
        log_rhk: log of chromospheric Ca HK flux ratio R'HK (e.g. -4.5 for Sun)
        bv_color: B-V colour index (used for spectral-type validity check)
    """
    if log_rhk > -3.5 or log_rhk < -6.0:
        return ChromosphericActivityResult(
            log_rhk=log_rhk,
            age_gyr=float("nan"),
            age_uncertainty_gyr=float("nan"),
            rv_jitter_ms=float("nan"),
            activity_level="UNKNOWN",
            flag="INVALID_LOG_RHK",
        )

    log_age = -38.053 - 17.912 * log_rhk - 1.6675 * log_rhk**2
    age_gyr = 10.0**log_age / 1e9

    # Uncertainty from relation scatter ~0.05 dex in log age
    age_lo = 10.0 ** (log_age - 0.5) / 1e9
    age_hi = 10.0 ** (log_age + 0.5) / 1e9
    age_unc = (age_hi - age_lo) / 2.0

    # RV jitter from Isaacson & Fischer (2010) empirical relation
    log_jitter = 0.51 * log_rhk + 4.72
    rv_jitter = max(0.5, 10.0**log_jitter)

    if log_rhk > -4.2:
        activity = "VERY_ACTIVE"
    elif log_rhk > -4.5:
        activity = "ACTIVE"
    elif log_rhk > -4.75:
        activity = "MODERATE"
    elif log_rhk > -5.0:
        activity = "QUIET"
    else:
        activity = "VERY_QUIET"

    return ChromosphericActivityResult(
        log_rhk=log_rhk,
        age_gyr=age_gyr,
        age_uncertainty_gyr=age_unc,
        rv_jitter_ms=rv_jitter,
        activity_level=activity,
        flag="OK",
    )


def format_chromospheric_activity_result(r: ChromosphericActivityResult) -> str:
    if r.flag != "OK":
        return f"ChromosphericActivity | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| log R'HK | {r.log_rhk:.3f} |\n"
        f"| Stellar age | {r.age_gyr:.2f} ± {r.age_uncertainty_gyr:.2f} Gyr |\n"
        f"| RV jitter | {r.rv_jitter_ms:.2f} m/s |\n"
        f"| Activity level | {r.activity_level} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Chromospheric activity age estimator")
    p.add_argument("log_rhk", type=float, help="log R'HK (e.g. -4.5 for Sun)")
    p.add_argument("--bv", type=float, default=0.65, help="B-V colour index")
    args = p.parse_args()
    r = compute_chromospheric_age(args.log_rhk, bv_color=args.bv)
    print(format_chromospheric_activity_result(r))


if __name__ == "__main__":
    _cli()
