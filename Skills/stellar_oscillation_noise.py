"""Estimate solar-like oscillation amplitude and granulation noise."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Kjeldsen & Bedding (1995) / Chaplin et al. (2011) scaling relations
_LSUN = 3.828e26    # W
_MSUN = 1.989e30    # kg
_TSUN = 5778.0      # K
_NU_MAX_SUN = 3090.0  # μHz  (peak oscillation frequency for the Sun)
_DELTA_NU_SUN = 135.1  # μHz  (large frequency separation for the Sun)

# Oscillation amplitude scales as: A ~ (L/Lsun)^s / (M/Msun)^t * (Teff/Tsun)^u
# Harvey model granulation: sigma_gran ~ C * nu_max^(-1/2)


@dataclass(frozen=True)
class OscillationNoiseResult:
    nu_max_uhz: float
    delta_nu_uhz: float
    oscillation_amplitude_ppm: float
    granulation_amplitude_ppm: float
    total_noise_ppm: float
    flag: str


def estimate_oscillation_noise(
    teff_k: float,
    luminosity_lsun: float | None = None,
    mass_msun: float = 1.0,
    logg: float | None = None,
) -> OscillationNoiseResult:
    """
    Estimate solar-like oscillation amplitude and granulation noise for a star.

    Uses Kjeldsen & Bedding (1995) scaling:
    - nu_max ~ (M/Msun) * (Teff/Tsun)^(-1/2) * nu_max_sun / (L/Lsun)^(3/4)
    - delta_nu ~ sqrt(M/Msun) * (L/Lsun)^(-3/4) * delta_nu_sun * (Teff/Tsun)^(3/8)
    - A_osc ~ (L/Lsun)^0.838 / (M/Msun) / (Teff/Tsun)^1.32  [ppm]
    - A_gran ~ 2 * (nu_max / nu_max_sun)^(-0.5) * 0.5        [ppm, Harvey background]

    logg can substitute for luminosity when L is unknown.
    """
    if not math.isfinite(teff_k) or teff_k <= 0.0:
        return OscillationNoiseResult(
            nu_max_uhz=float("nan"), delta_nu_uhz=float("nan"),
            oscillation_amplitude_ppm=float("nan"),
            granulation_amplitude_ppm=float("nan"),
            total_noise_ppm=float("nan"), flag="INVALID_TEFF",
        )
    if not math.isfinite(mass_msun) or mass_msun <= 0.0:
        return OscillationNoiseResult(
            nu_max_uhz=float("nan"), delta_nu_uhz=float("nan"),
            oscillation_amplitude_ppm=float("nan"),
            granulation_amplitude_ppm=float("nan"),
            total_noise_ppm=float("nan"), flag="INVALID_MASS",
        )

    # Derive luminosity from logg if not provided
    if luminosity_lsun is None or not math.isfinite(luminosity_lsun):
        if logg is not None and math.isfinite(logg):
            g_si = (10 ** logg) / 100.0
            r_m = math.sqrt(6.674e-11 * mass_msun * _MSUN / g_si)
            r_rsun = r_m / 6.957e8
            luminosity_lsun = r_rsun**2 * (teff_k / _TSUN) ** 4
        else:
            # Approximate from main-sequence mass-luminosity
            luminosity_lsun = mass_msun ** 4.0

    lum = max(luminosity_lsun, 1e-4)

    teff_ratio = teff_k / _TSUN

    # nu_max scaling (Brown et al. 1991)
    nu_max = _NU_MAX_SUN * (mass_msun / lum) * teff_ratio ** (-0.5)

    # delta_nu scaling
    delta_nu = _DELTA_NU_SUN * math.sqrt(mass_msun) * lum ** (-0.75) * teff_ratio ** (3.0 / 8.0)

    # Oscillation amplitude (Chaplin et al. 2011 simplified)
    a_osc = 3.6 * (lum ** 0.838) / mass_msun / (teff_ratio ** 1.32)

    # Granulation amplitude (Harvey model, ~0.5 ppm for solar)
    a_gran = 0.5 * math.sqrt(_NU_MAX_SUN / nu_max) if nu_max > 0 else float("nan")

    total = math.sqrt(a_osc**2 + a_gran**2) if math.isfinite(a_gran) else a_osc

    return OscillationNoiseResult(
        nu_max_uhz=round(nu_max, 2),
        delta_nu_uhz=round(delta_nu, 3),
        oscillation_amplitude_ppm=round(a_osc, 3),
        granulation_amplitude_ppm=round(a_gran, 3) if math.isfinite(a_gran) else float("nan"),
        total_noise_ppm=round(total, 3),
        flag="OK",
    )


def format_oscillation_noise_result(r: OscillationNoiseResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| ν_max (μHz) | {_f(r.nu_max_uhz)} |\n"
        f"| Δν (μHz) | {_f(r.delta_nu_uhz)} |\n"
        f"| Oscillation amplitude (ppm) | {_f(r.oscillation_amplitude_ppm)} |\n"
        f"| Granulation amplitude (ppm) | {_f(r.granulation_amplitude_ppm)} |\n"
        f"| Total noise (ppm) | {_f(r.total_noise_ppm)} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate solar-like oscillation noise.")
    p.add_argument("teff_k", type=float)
    p.add_argument("--luminosity-lsun", type=float, default=None)
    p.add_argument("--mass-msun", type=float, default=1.0)
    p.add_argument("--logg", type=float, default=None)
    args = p.parse_args()
    r = estimate_oscillation_noise(args.teff_k, args.luminosity_lsun, args.mass_msun, args.logg)
    print(format_oscillation_noise_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
