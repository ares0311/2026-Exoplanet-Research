"""Estimate sky background brightness from lunar conditions."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SkyBrightnessResult:
    moon_illumination: float
    moon_separation_deg: float
    sky_brightness_vmag: float
    dark_sky_vmag: float
    excess_mag: float
    flag: str


# Dark sky zenith: 22.0 mag/arcsec² (V-band)
_DARK_SKY_VMAG = 22.0


def estimate_sky_brightness(
    moon_illumination: float,
    moon_separation_deg: float,
    dark_sky_vmag: float = _DARK_SKY_VMAG,
) -> SkyBrightnessResult:
    """
    Estimate V-band sky surface brightness (mag/arcsec²) from lunar illumination
    and angular separation, using the Krisciunas & Schaefer (1991) approximation.
    """
    if not math.isfinite(moon_illumination) or not (0.0 <= moon_illumination <= 1.0):
        return SkyBrightnessResult(
            moon_illumination=moon_illumination,
            moon_separation_deg=moon_separation_deg,
            sky_brightness_vmag=float("nan"),
            dark_sky_vmag=dark_sky_vmag,
            excess_mag=float("nan"),
            flag="INVALID_ILLUMINATION",
        )
    if not math.isfinite(moon_separation_deg) or moon_separation_deg < 0.0:
        return SkyBrightnessResult(
            moon_illumination=moon_illumination,
            moon_separation_deg=moon_separation_deg,
            sky_brightness_vmag=float("nan"),
            dark_sky_vmag=dark_sky_vmag,
            excess_mag=float("nan"),
            flag="INVALID_SEPARATION",
        )

    # Moon V magnitude from illumination (full moon ≈ -12.74)
    if moon_illumination <= 0.0:
        # New moon — dark sky
        sky_vmag = dark_sky_vmag
        return SkyBrightnessResult(
            moon_illumination=moon_illumination,
            moon_separation_deg=moon_separation_deg,
            sky_brightness_vmag=round(sky_vmag, 3),
            dark_sky_vmag=dark_sky_vmag,
            excess_mag=0.0,
            flag="OK",
        )

    moon_vmag = -12.74 + 2.5 * math.log10(1.0 / moon_illumination)

    # Lunar scattering function — simplified angular term
    rho = max(moon_separation_deg, 5.0)
    rho_rad = math.radians(rho)
    f_rho = (10 ** 5.36) * (1.06 + math.cos(rho_rad) ** 2) + 10 ** (6.15 - rho / 40.0)

    # Moon illuminance at telescope (nL)
    i_moon = f_rho * 10 ** (-0.4 * (moon_vmag + 16.57))

    # Dark sky brightness in nL (22.0 mag/arcsec² ≈ 0.171 nL/arcsec²)
    b_dark_nl = 34.08 * math.exp(20.7233 - 0.92104 * dark_sky_vmag)

    b_total_nl = b_dark_nl + i_moon
    sky_vmag = (20.7233 - math.log(b_total_nl / 34.08)) / 0.92104
    excess_mag = dark_sky_vmag - sky_vmag

    return SkyBrightnessResult(
        moon_illumination=moon_illumination,
        moon_separation_deg=moon_separation_deg,
        sky_brightness_vmag=round(sky_vmag, 3),
        dark_sky_vmag=dark_sky_vmag,
        excess_mag=round(excess_mag, 3),
        flag="OK",
    )


def format_sky_brightness_result(r: SkyBrightnessResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Moon illumination | {r.moon_illumination:.2f} |\n"
        f"| Moon separation (°) | {r.moon_separation_deg:.1f} |\n"
        f"| Sky brightness (V mag/arcsec²) | {r.sky_brightness_vmag:.3f} |\n"
        f"| Dark sky reference | {r.dark_sky_vmag:.3f} |\n"
        f"| Excess (mag) | {r.excess_mag:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate sky brightness from lunar conditions.")
    p.add_argument("moon_illumination", type=float, help="Moon illumination fraction 0–1")
    p.add_argument("moon_separation_deg", type=float, help="Moon-target separation in degrees")
    p.add_argument("--dark-sky", type=float, default=_DARK_SKY_VMAG)
    args = p.parse_args()
    r = estimate_sky_brightness(args.moon_illumination, args.moon_separation_deg, args.dark_sky)
    print(format_sky_brightness_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
