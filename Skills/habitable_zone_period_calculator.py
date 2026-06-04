"""Calculate orbital periods at habitable zone boundaries from stellar luminosity."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_AU_M = 1.495978707e11
_SEC_PER_DAY = 86400.0


@dataclass(frozen=True)
class HabitableZonePeriodResult:
    stellar_luminosity_lsun: float
    hz_inner_au: float
    hz_outer_au: float
    hz_inner_period_days: float
    hz_outer_period_days: float
    hz_optimistic_inner_au: float
    hz_optimistic_outer_au: float
    hz_optimistic_inner_period_days: float
    hz_optimistic_outer_period_days: float
    flag: str


def compute_hz_periods(
    stellar_luminosity_lsun: float,
    stellar_mass_msun: float = 1.0,
) -> HabitableZonePeriodResult:
    """Compute orbital periods at conservative and optimistic HZ boundaries.

    Uses Kopparapu et al. (2013) flux limits scaled to solar:
      Conservative: S_inner = 1.107 (moist greenhouse), S_outer = 0.356 (max greenhouse)
      Optimistic:   S_inner = 1.776 (recent Venus),     S_outer = 0.320 (early Mars)

    a = sqrt(L / S_eff), then Kepler's 3rd law P = 2π√(a³/GM).

    Args:
        stellar_luminosity_lsun: stellar luminosity (solar luminosities)
        stellar_mass_msun: stellar mass (solar masses), used for Kepler's 3rd law
    """
    if stellar_luminosity_lsun <= 0.0:
        return HabitableZonePeriodResult(
            stellar_luminosity_lsun=stellar_luminosity_lsun,
            hz_inner_au=float("nan"), hz_outer_au=float("nan"),
            hz_inner_period_days=float("nan"), hz_outer_period_days=float("nan"),
            hz_optimistic_inner_au=float("nan"), hz_optimistic_outer_au=float("nan"),
            hz_optimistic_inner_period_days=float("nan"),
            hz_optimistic_outer_period_days=float("nan"),
            flag="INVALID_LUMINOSITY",
        )
    if stellar_mass_msun <= 0.0:
        return HabitableZonePeriodResult(
            stellar_luminosity_lsun=stellar_luminosity_lsun,
            hz_inner_au=float("nan"), hz_outer_au=float("nan"),
            hz_inner_period_days=float("nan"), hz_outer_period_days=float("nan"),
            hz_optimistic_inner_au=float("nan"), hz_optimistic_outer_au=float("nan"),
            hz_optimistic_inner_period_days=float("nan"),
            hz_optimistic_outer_period_days=float("nan"),
            flag="INVALID_STELLAR_MASS",
        )

    s_inner = 1.107
    s_outer = 0.356
    s_opt_inner = 1.776
    s_opt_outer = 0.320

    def au_from_flux(s_eff: float) -> float:
        return math.sqrt(stellar_luminosity_lsun / s_eff)

    def period_days(a_au: float) -> float:
        a_m = a_au * _AU_M
        ms_kg = stellar_mass_msun * _MSUN_KG
        return 2.0 * math.pi * math.sqrt(a_m**3 / (_G * ms_kg)) / _SEC_PER_DAY

    hz_in = au_from_flux(s_inner)
    hz_out = au_from_flux(s_outer)
    hz_opt_in = au_from_flux(s_opt_inner)
    hz_opt_out = au_from_flux(s_opt_outer)

    return HabitableZonePeriodResult(
        stellar_luminosity_lsun=stellar_luminosity_lsun,
        hz_inner_au=hz_in,
        hz_outer_au=hz_out,
        hz_inner_period_days=period_days(hz_in),
        hz_outer_period_days=period_days(hz_out),
        hz_optimistic_inner_au=hz_opt_in,
        hz_optimistic_outer_au=hz_opt_out,
        hz_optimistic_inner_period_days=period_days(hz_opt_in),
        hz_optimistic_outer_period_days=period_days(hz_opt_out),
        flag="OK",
    )


def format_hz_period_result(r: HabitableZonePeriodResult) -> str:
    if r.flag != "OK":
        return f"HabitableZonePeriod | flag={r.flag}"
    return (
        f"| Boundary | Distance (AU) | Period (days) |\n"
        f"|---|---|---|\n"
        f"| Conservative inner (moist GH) | {r.hz_inner_au:.3f} | {r.hz_inner_period_days:.1f} |\n"
        f"| Conservative outer (max GH) | {r.hz_outer_au:.3f} | {r.hz_outer_period_days:.1f} |\n"
        f"| Optimistic inner (recent Venus) | {r.hz_optimistic_inner_au:.3f}"
        f" | {r.hz_optimistic_inner_period_days:.1f} |\n"
        f"| Optimistic outer (early Mars) | {r.hz_optimistic_outer_au:.3f}"
        f" | {r.hz_optimistic_outer_period_days:.1f} |\n"
        f"| Luminosity | {r.stellar_luminosity_lsun:.4f} L☉ | |\n"
        f"| Flag | {r.flag} | |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Habitable zone orbital period calculator")
    p.add_argument("luminosity_lsun", type=float, help="Stellar luminosity (L_sun)")
    p.add_argument("--mstar", type=float, default=1.0, help="Stellar mass (M_sun)")
    args = p.parse_args()
    r = compute_hz_periods(args.luminosity_lsun, stellar_mass_msun=args.mstar)
    print(format_hz_period_result(r))


if __name__ == "__main__":
    _cli()
