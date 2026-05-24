"""Find the optimal photometric aperture radius to maximise transit SNR.

Uses a Gaussian PSF enclosed-energy model to estimate signal fraction and
noise contributions (target photon noise + sky + neighbour contamination)
as a function of aperture radius.  Distinct from ``crowding_metric_calculator``
(which reports a single contamination metric) — this scores candidate radii
and returns the optimal one.

Public API
----------
ApertureScore(radius_arcsec, signal_fraction, noise_fraction,
              snr_proxy, contamination_fraction)
ApertureOptResult(target_tmag, psf_fwhm_arcsec, candidate_radii_arcsec,
                  scores, optimal_radius_arcsec, optimal_snr_proxy,
                  flag)
score_apertures(target_tmag, neighbor_tmags, neighbor_separations_arcsec, *,
                psf_fwhm_arcsec, candidate_radii_arcsec,
                sky_noise_per_pixel, pixel_scale_arcsec) -> ApertureOptResult
format_aperture_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ApertureScore:
    radius_arcsec: float
    signal_fraction: float        # EE of target within aperture
    noise_fraction: float         # sqrt(signal + sky_pix + contamination) proxy
    snr_proxy: float              # signal_fraction / noise_fraction
    contamination_fraction: float # neighbour flux / (target + neighbour) within aperture


@dataclass(frozen=True)
class ApertureOptResult:
    target_tmag: float
    psf_fwhm_arcsec: float
    candidate_radii_arcsec: tuple[float, ...]
    scores: tuple[ApertureScore, ...]
    optimal_radius_arcsec: float | None
    optimal_snr_proxy: float | None
    flag: str  # "OK" | "INVALID" | "INSUFFICIENT"


def _gaussian_ee(radius: float, fwhm: float) -> float:
    """Enclosed energy of a Gaussian PSF within *radius* (same units as *fwhm*)."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return 1.0 - math.exp(-0.5 * (radius / sigma) ** 2)


def _flux_from_mag(mag: float) -> float:
    """Relative flux (arbitrary units) from magnitude."""
    return 10.0 ** (-0.4 * mag)


def score_apertures(
    target_tmag: float,
    neighbor_tmags: list[float],
    neighbor_separations_arcsec: list[float],
    *,
    psf_fwhm_arcsec: float = 2.0,
    candidate_radii_arcsec: list[float] | None = None,
    sky_noise_per_pixel: float = 1.0,
    pixel_scale_arcsec: float = 21.0,
) -> ApertureOptResult:
    """Score candidate aperture radii and return the one with highest SNR proxy.

    Args:
        target_tmag: TESS magnitude of the target star.
        neighbor_tmags: TESS magnitudes of nearby stars.
        neighbor_separations_arcsec: Angular separations (arcsec) of neighbours.
        psf_fwhm_arcsec: PSF FWHM (arcsec).
        candidate_radii_arcsec: Aperture radii to evaluate.  Defaults to
            0.5–4.0 × FWHM in 8 steps.
        sky_noise_per_pixel: Sky background noise per pixel (arbitrary units).
        pixel_scale_arcsec: Plate scale (arcsec/pixel).

    Returns:
        :class:`ApertureOptResult`.
    """
    if not math.isfinite(target_tmag) or psf_fwhm_arcsec <= 0 or pixel_scale_arcsec <= 0:
        return ApertureOptResult(target_tmag, psf_fwhm_arcsec, (), (), None, None, "INVALID")
    if len(neighbor_tmags) != len(neighbor_separations_arcsec):
        return ApertureOptResult(target_tmag, psf_fwhm_arcsec, (), (), None, None, "INVALID")
    if any(s < 0 for s in neighbor_separations_arcsec):
        return ApertureOptResult(target_tmag, psf_fwhm_arcsec, (), (), None, None, "INVALID")

    if candidate_radii_arcsec is None:
        candidate_radii_arcsec = [
            psf_fwhm_arcsec * f for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
        ]
    if not candidate_radii_arcsec:
        return ApertureOptResult(
            target_tmag, psf_fwhm_arcsec,
            tuple(candidate_radii_arcsec), (), None, None, "INSUFFICIENT"
        )

    target_flux = _flux_from_mag(target_tmag)
    neighbour_fluxes = [_flux_from_mag(m) for m in neighbor_tmags]

    scores: list[ApertureScore] = []
    for r in candidate_radii_arcsec:
        if r <= 0:
            continue
        sig_frac = _gaussian_ee(r, psf_fwhm_arcsec)
        target_within = target_flux * sig_frac

        # Sky noise scales with aperture area
        n_pixels = math.pi * (r / pixel_scale_arcsec) ** 2
        sky_var = sky_noise_per_pixel ** 2 * n_pixels

        # Neighbour flux within aperture — shift PSF by separation
        neigh_within = sum(
            f * _gaussian_ee(max(0.0, r - sep), psf_fwhm_arcsec)
            if sep < r else
            f * _gaussian_ee(r, psf_fwhm_arcsec) * math.exp(-0.5 * (sep / psf_fwhm_arcsec) ** 2)
            for f, sep in zip(neighbour_fluxes, neighbor_separations_arcsec, strict=False)
        )

        total_within = target_within + neigh_within
        noise = math.sqrt(max(total_within, 1e-30) + sky_var)
        snr_proxy = target_within / noise if noise > 0 else 0.0
        cont = neigh_within / total_within if total_within > 0 else 0.0

        scores.append(ApertureScore(
            radius_arcsec=round(r, 4),
            signal_fraction=round(sig_frac, 6),
            noise_fraction=round(noise, 6),
            snr_proxy=round(snr_proxy, 6),
            contamination_fraction=round(cont, 6),
        ))

    if not scores:
        return ApertureOptResult(
            target_tmag, psf_fwhm_arcsec,
            tuple(candidate_radii_arcsec), (), None, None, "INSUFFICIENT"
        )

    best = max(scores, key=lambda s: s.snr_proxy)
    return ApertureOptResult(
        target_tmag=target_tmag,
        psf_fwhm_arcsec=psf_fwhm_arcsec,
        candidate_radii_arcsec=tuple(round(r, 4) for r in candidate_radii_arcsec),
        scores=tuple(scores),
        optimal_radius_arcsec=best.radius_arcsec,
        optimal_snr_proxy=best.snr_proxy,
        flag="OK",
    )


def format_aperture_result(result: ApertureOptResult) -> str:
    """Format aperture optimisation result as Markdown."""
    lines = [
        "## Aperture Optimization Scorer",
        "",
        f"- Target Tmag: {result.target_tmag}",
        f"- PSF FWHM: {result.psf_fwhm_arcsec} arcsec",
        f"- **Optimal aperture: {result.optimal_radius_arcsec} arcsec**",
        f"- **Optimal SNR proxy: {result.optimal_snr_proxy}**",
        f"- **Flag: {result.flag}**",
    ]
    if result.scores:
        lines += ["", "| Radius (arcsec) | EE | SNR proxy | Contamination |", "|---|---|---|---|"]
        for s in result.scores:
            lines.append(
                f"| {s.radius_arcsec} | {s.signal_fraction:.4f}"
                f" | {s.snr_proxy:.4f} | {s.contamination_fraction:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="aperture_optimization_scorer",
        description="Find optimal aperture radius for transit photometry.",
    )
    parser.add_argument("target_tmag", type=float)
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = score_apertures(args.target_tmag, [], [], psf_fwhm_arcsec=args.psf_fwhm)
    print(format_aperture_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
