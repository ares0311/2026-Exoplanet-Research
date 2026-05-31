"""Check if a planet's radius exceeds its Roche lobe (EB diagnostic)."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_REARTH_AU = 4.2635e-5
_MEARTH_MSUN = 3.003e-6


@dataclass(frozen=True)
class RocheLobResult:
    rp_rearth: float
    mp_mearth: float
    ms_msun: float
    sma_au: float
    roche_radius_rearth: float
    hill_radius_rearth: float
    fill_factor: float
    eb_suspected: bool
    flag: str


def check_roche_lobe(
    rp_rearth: float,
    mp_mearth: float,
    ms_msun: float,
    sma_au: float,
) -> RocheLobResult:
    """Compute Roche lobe and Hill sphere radii; flag as suspected EB if fill > 0.5."""
    if not math.isfinite(rp_rearth) or rp_rearth <= 0.0:
        return RocheLobResult(
            rp_rearth=rp_rearth, mp_mearth=mp_mearth, ms_msun=ms_msun, sma_au=sma_au,
            roche_radius_rearth=float("nan"), hill_radius_rearth=float("nan"),
            fill_factor=float("nan"), eb_suspected=False, flag="INVALID_PLANET_RADIUS",
        )
    if not math.isfinite(mp_mearth) or mp_mearth <= 0.0:
        return RocheLobResult(
            rp_rearth=rp_rearth, mp_mearth=mp_mearth, ms_msun=ms_msun, sma_au=sma_au,
            roche_radius_rearth=float("nan"), hill_radius_rearth=float("nan"),
            fill_factor=float("nan"), eb_suspected=False, flag="INVALID_PLANET_MASS",
        )
    if not math.isfinite(ms_msun) or ms_msun <= 0.0:
        return RocheLobResult(
            rp_rearth=rp_rearth, mp_mearth=mp_mearth, ms_msun=ms_msun, sma_au=sma_au,
            roche_radius_rearth=float("nan"), hill_radius_rearth=float("nan"),
            fill_factor=float("nan"), eb_suspected=False, flag="INVALID_STELLAR_MASS",
        )
    if not math.isfinite(sma_au) or sma_au <= 0.0:
        return RocheLobResult(
            rp_rearth=rp_rearth, mp_mearth=mp_mearth, ms_msun=ms_msun, sma_au=sma_au,
            roche_radius_rearth=float("nan"), hill_radius_rearth=float("nan"),
            fill_factor=float("nan"), eb_suspected=False, flag="INVALID_SMA",
        )

    q = (mp_mearth * _MEARTH_MSUN) / ms_msun
    sma_rearth = sma_au / _REARTH_AU

    # Eggleton (1983) Roche lobe approximation
    q13 = q ** (1.0 / 3.0)
    q23 = q ** (2.0 / 3.0)
    roche_frac = 0.49 * q23 / (0.6 * q23 + math.log(1.0 + q13))
    roche_rearth = roche_frac * sma_rearth

    # Hill sphere
    hill_rearth = sma_rearth * (q / 3.0) ** (1.0 / 3.0)

    fill_factor = rp_rearth / roche_rearth
    eb_suspected = fill_factor > 0.5

    return RocheLobResult(
        rp_rearth=rp_rearth,
        mp_mearth=mp_mearth,
        ms_msun=ms_msun,
        sma_au=sma_au,
        roche_radius_rearth=round(roche_rearth, 4),
        hill_radius_rearth=round(hill_rearth, 4),
        fill_factor=round(fill_factor, 6),
        eb_suspected=eb_suspected,
        flag="EB_SUSPECTED" if eb_suspected else "OK",
    )


def format_roche_lob_result(r: RocheLobResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Rp (R⊕) | {r.rp_rearth:.2f} |\n"
        f"| Roche radius (R⊕) | {r.roche_radius_rearth:.2f} |\n"
        f"| Hill radius (R⊕) | {r.hill_radius_rearth:.2f} |\n"
        f"| Fill factor | {r.fill_factor:.4f} |\n"
        f"| EB suspected | {r.eb_suspected} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check Roche lobe fill factor.")
    p.add_argument("rp_rearth", type=float)
    p.add_argument("mp_mearth", type=float)
    p.add_argument("ms_msun", type=float)
    p.add_argument("sma_au", type=float)
    args = p.parse_args()
    r = check_roche_lobe(args.rp_rearth, args.mp_mearth, args.ms_msun, args.sma_au)
    print(format_roche_lob_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
