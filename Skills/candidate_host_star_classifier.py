"""Classify host star type from effective temperature, surface gravity, and rotation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostStarClassificationResult:
    spectral_type: str          # OBAFGKM + subtype estimate
    luminosity_class: str       # MAIN_SEQUENCE / SUBGIANT / GIANT / SUPERGIANT
    stellar_class: str          # combined e.g. "G2V"
    is_pulsator_risk: bool      # delta Scuti / gamma Dor instability strip
    is_giant_risk: bool         # diluted transit concern
    tidal_circularization_risk: bool  # close orbit tidal effects likely
    flag: str


def classify_host_star(
    teff_k: float,
    logg: float,
    vsini_kms: float = 0.0,
) -> HostStarClassificationResult:
    """Classify a host star from Teff, log g, and v*sin(i).

    Spectral type from Pecaut & Mamajek (2013) Teff scale.
    Luminosity class from log g thresholds.
    Pulsator risk: delta Scuti (6500–8000 K, logg>3.5), gamma Dor (6800–7800 K).
    Giant risk: logg < 3.5.

    Args:
        teff_k: stellar effective temperature (K)
        logg: log surface gravity (cgs)
        vsini_kms: projected rotational velocity (km/s); used for pulsator assessment
    """
    if teff_k <= 0.0:
        return HostStarClassificationResult(
            spectral_type="UNKNOWN", luminosity_class="UNKNOWN",
            stellar_class="UNKNOWN", is_pulsator_risk=False,
            is_giant_risk=False, tidal_circularization_risk=False,
            flag="INVALID_TEFF",
        )
    if logg < 0.0 or logg > 6.0:
        return HostStarClassificationResult(
            spectral_type="UNKNOWN", luminosity_class="UNKNOWN",
            stellar_class="UNKNOWN", is_pulsator_risk=False,
            is_giant_risk=False, tidal_circularization_risk=False,
            flag="INVALID_LOGG",
        )

    # Spectral type: Pecaut & Mamajek (2013) approximate boundaries
    if teff_k >= 30000:
        sp = "O"
    elif teff_k >= 10000:
        sp = "B"
    elif teff_k >= 7500:
        sp = "A"
    elif teff_k >= 6000:
        sp = "F"
    elif teff_k >= 5200:
        sp = "G"
    elif teff_k >= 3700:
        sp = "K"
    else:
        sp = "M"

    # Subtype 0–9 from Teff within class
    _class_bounds = {
        "O": (30000, 50000), "B": (10000, 30000), "A": (7500, 10000),
        "F": (6000, 7500), "G": (5200, 6000), "K": (3700, 5200), "M": (2400, 3700),
    }
    lo, hi = _class_bounds[sp]
    subtype = int(9 * (1.0 - (teff_k - lo) / (hi - lo)))
    subtype = max(0, min(9, subtype))

    # Luminosity class from log g
    if logg >= 4.0:
        lum_class = "MAIN_SEQUENCE"
        lum_suffix = "V"
    elif logg >= 3.5:
        lum_class = "SUBGIANT"
        lum_suffix = "IV"
    elif logg >= 2.5:
        lum_class = "GIANT"
        lum_suffix = "III"
    else:
        lum_class = "SUPERGIANT"
        lum_suffix = "I"

    stellar_class = f"{sp}{subtype}{lum_suffix}"

    # Delta Scuti pulsator instability strip: 6500–8000 K, main sequence
    is_pulsator = (6500 <= teff_k <= 8000) and logg >= 3.5

    is_giant = logg < 3.5
    # Tidal risk: rapid rotator (vsini > 20 km/s) suggests short period → tidal effects
    tidal_risk = vsini_kms > 20.0

    return HostStarClassificationResult(
        spectral_type=f"{sp}{subtype}",
        luminosity_class=lum_class,
        stellar_class=stellar_class,
        is_pulsator_risk=is_pulsator,
        is_giant_risk=is_giant,
        tidal_circularization_risk=tidal_risk,
        flag="OK",
    )


def format_host_star_classification_result(r: HostStarClassificationResult) -> str:
    if r.flag != "OK":
        return f"HostStarClassification | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Spectral type | {r.spectral_type} |\n"
        f"| Luminosity class | {r.luminosity_class} |\n"
        f"| Full stellar class | {r.stellar_class} |\n"
        f"| Pulsator risk | {'YES' if r.is_pulsator_risk else 'no'} |\n"
        f"| Giant/dilution risk | {'YES' if r.is_giant_risk else 'no'} |\n"
        f"| Tidal circularization risk | {'YES' if r.tidal_circularization_risk else 'no'} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Candidate host star classifier")
    p.add_argument("teff_k", type=float, help="Stellar Teff (K)")
    p.add_argument("logg", type=float, help="log g (cgs)")
    p.add_argument("--vsini", type=float, default=0.0, help="v*sin(i) (km/s)")
    args = p.parse_args()
    r = classify_host_star(args.teff_k, args.logg, vsini_kms=args.vsini)
    print(format_host_star_classification_result(r))


if __name__ == "__main__":
    _cli()
