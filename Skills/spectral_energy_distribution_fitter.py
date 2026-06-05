"""Fit stellar effective temperature from multi-band photometry via blackbody SED."""
from __future__ import annotations

import math
from dataclasses import dataclass

_H = 6.626e-34
_C = 2.998e8
_K_B = 1.381e-23

# Approximate effective wavelengths in microns for common bands
_BAND_WAVELENGTHS_UM = {
    "U": 0.365, "B": 0.440, "V": 0.550, "R": 0.640, "I": 0.800,
    "J": 1.235, "H": 1.662, "K": 2.159, "W1": 3.368, "W2": 4.618,
    "W3": 12.082, "W4": 22.194,
}


def _planck_nu(wavelength_um: float, teff_k: float) -> float:
    wl_m = wavelength_um * 1e-6
    x = _H * _C / (_K_B * teff_k * wl_m)
    if x > 700.0:
        return 0.0
    return (2.0 * _H * _C**2 / wl_m**5) / (math.exp(x) - 1.0)


@dataclass(frozen=True)
class SedFitResult:
    fitted_teff_k: float
    best_chi2: float
    n_bands: int
    teff_uncertainty_k: float
    flag: str


def fit_sed_temperature(
    band_magnitudes: dict[str, float],
    teff_grid_k: list[float] | None = None,
) -> SedFitResult:
    """Fit stellar Teff by minimising χ² between observed and blackbody flux ratios.

    Uses flux ratios between bands to eliminate the normalisation uncertainty.
    Grid search over Teff from 2500 K to 50000 K.

    Args:
        band_magnitudes: dict of band name → apparent magnitude (e.g. {"B":10.2, "V":9.8})
        teff_grid_k: optional explicit temperature grid; defaults to 100K steps 2500–50000K
    """
    if len(band_magnitudes) < 2:
        return SedFitResult(float("nan"), float("nan"), len(band_magnitudes),
                             float("nan"), "INSUFFICIENT_BANDS")

    bands = []
    for band, mag in band_magnitudes.items():
        wl = _BAND_WAVELENGTHS_UM.get(band)
        if wl is None:
            continue
        bands.append((wl, mag))

    if len(bands) < 2:
        return SedFitResult(float("nan"), float("nan"), len(bands),
                             float("nan"), "UNKNOWN_BANDS")

    if teff_grid_k is None:
        teff_grid_k = list(range(2500, 50001, 100))

    wls = [b[0] for b in bands]
    mags = [b[1] for b in bands]

    best_teff = float("nan")
    best_chi2 = float("inf")

    for teff in teff_grid_k:
        fluxes = [_planck_nu(wl, teff) for wl in wls]
        if any(f <= 0.0 for f in fluxes):
            continue
        model_mags = [-2.5 * math.log10(f) for f in fluxes]
        # Fit offset: normalise to mean
        offset = sum(mags[i] - model_mags[i] for i in range(len(bands))) / len(bands)
        chi2 = sum((mags[i] - (model_mags[i] + offset)) ** 2 for i in range(len(bands)))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_teff = float(teff)

    if math.isnan(best_teff):
        return SedFitResult(float("nan"), float("nan"), len(bands),
                             float("nan"), "FIT_FAILED")

    # Uncertainty: Δchi2 = 1 → range of acceptable Teff
    teff_lo = best_teff
    teff_hi = best_teff
    for teff in teff_grid_k:
        fluxes = [_planck_nu(wl, teff) for wl in wls]
        if any(f <= 0.0 for f in fluxes):
            continue
        model_mags = [-2.5 * math.log10(f) for f in fluxes]
        offset = sum(mags[i] - model_mags[i] for i in range(len(bands))) / len(bands)
        chi2 = sum((mags[i] - (model_mags[i] + offset)) ** 2 for i in range(len(bands)))
        if chi2 <= best_chi2 + 1.0:
            teff_lo = min(teff_lo, float(teff))
            teff_hi = max(teff_hi, float(teff))

    teff_unc = (teff_hi - teff_lo) / 2.0

    return SedFitResult(
        fitted_teff_k=best_teff,
        best_chi2=best_chi2,
        n_bands=len(bands),
        teff_uncertainty_k=max(teff_unc, 50.0),
        flag="OK",
    )


def format_sed_fit_result(r: SedFitResult) -> str:
    if r.flag != "OK":
        return f"SedFit | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Fitted Teff | {r.fitted_teff_k:.0f} ± {r.teff_uncertainty_k:.0f} K |\n"
        f"| Best χ² | {r.best_chi2:.4f} |\n"
        f"| Bands used | {r.n_bands} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Blackbody SED temperature fitter")
    p.add_argument("--band", action="append", nargs=2, metavar=("BAND", "MAG"),
                   required=True, help="Band and magnitude, e.g. --band B 10.2 --band V 9.8")
    args = p.parse_args()
    band_mags = {b: float(m) for b, m in args.band}
    r = fit_sed_temperature(band_mags)
    print(format_sed_fit_result(r))


if __name__ == "__main__":
    _cli()
