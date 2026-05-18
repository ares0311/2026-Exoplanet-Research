"""Estimate telescope time needed to observe a transit follow-up.

Converts V-magnitude to photon flux, then solves for the exposure time
required to reach a target SNR on the transit depth.

Public API
----------
TelescopeTimeResult(target_snr, depth_ppm, star_vmag, telescope_diameter_cm,
                    exposure_time_sec, n_exposures_in_transit, overhead_fraction,
                    total_time_hours, is_feasible, limiting_factor, flag)
estimate_telescope_time(depth_ppm, duration_hours, star_vmag,
                        telescope_diameter_cm, *, target_snr, ...) -> TelescopeTimeResult
format_telescope_time_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# V=0 photon flux: ~1000 photons/s/cm²/nm at 550 nm, integrated over ~200 nm → ~2e8 photons/s/cm²
_VBAND_ZERO_FLUX_PH_S_CM2 = 1.0e6  # photons/s/cm² at V=0 (effective for broadband)


@dataclass(frozen=True)
class TelescopeTimeResult:
    target_snr: float
    depth_ppm: float
    star_vmag: float
    telescope_diameter_cm: float
    exposure_time_sec: float
    n_exposures_in_transit: int
    overhead_fraction: float
    total_time_hours: float
    is_feasible: bool
    limiting_factor: str  # "photon_noise" | "readout" | "duration"
    flag: str             # "OK" | "TOO_FAINT" | "INVALID"


def estimate_telescope_time(
    depth_ppm: float,
    duration_hours: float,
    star_vmag: float,
    telescope_diameter_cm: float,
    *,
    target_snr: float = 10.0,
    overhead_fraction: float = 0.20,
    read_noise_e: float = 10.0,
    sky_background_e_per_s: float = 5.0,
    quantum_efficiency: float = 0.70,
    passband_bandwidth_nm: float = 200.0,
    max_hours: float = 8.0,
) -> TelescopeTimeResult:
    """Estimate telescope time for a transit follow-up.

    Args:
        depth_ppm: Transit depth in ppm.
        duration_hours: Transit duration in hours.
        star_vmag: Target V-band magnitude.
        telescope_diameter_cm: Telescope primary diameter in cm.
        target_snr: Required SNR on the transit depth detection.
        overhead_fraction: Fraction of time lost to readout/slewing.
        read_noise_e: Read noise in electrons per exposure.
        sky_background_e_per_s: Sky background in e⁻/s per pixel aperture.
        quantum_efficiency: Detector QE.
        passband_bandwidth_nm: Effective bandwidth in nm.
        max_hours: Maximum feasible total observing time.

    Returns:
        :class:`TelescopeTimeResult`.
    """
    if depth_ppm <= 0 or duration_hours <= 0 or telescope_diameter_cm <= 0:
        return TelescopeTimeResult(
            target_snr, depth_ppm, star_vmag, telescope_diameter_cm,
            0.0, 0, overhead_fraction, 0.0, False, "photon_noise", "INVALID",
        )

    area_cm2 = math.pi * (telescope_diameter_cm / 2.0) ** 2
    source_flux = (_VBAND_ZERO_FLUX_PH_S_CM2
                   * 10 ** (-0.4 * star_vmag)
                   * area_cm2
                   * quantum_efficiency
                   * passband_bandwidth_nm / 100.0)

    if source_flux < 1.0:
        return TelescopeTimeResult(
            target_snr, depth_ppm, star_vmag, telescope_diameter_cm,
            0.0, 0, overhead_fraction, 0.0, False, "photon_noise", "TOO_FAINT",
        )

    signal_fraction = depth_ppm / 1e6

    # Solve for t_exp: SNR = signal / noise
    # signal = source_flux * t * signal_fraction
    # noise = sqrt(source_flux*t + sky*t + read^2)
    # SNR^2 * (source*t + sky*t + read^2) = (source * signal_fraction)^2 * t^2
    # Quadratic in t: A*t^2 - B*t - C = 0
    a_coef = (source_flux * signal_fraction) ** 2
    b_coef = target_snr ** 2 * (source_flux + sky_background_e_per_s)
    c_coef = target_snr ** 2 * read_noise_e ** 2

    discriminant = b_coef ** 2 + 4 * a_coef * c_coef
    t_exp = (b_coef + math.sqrt(discriminant)) / (2 * a_coef)

    # Cap single exposure at transit duration
    transit_sec = duration_hours * 3600.0
    t_exp = min(t_exp, transit_sec)
    t_exp = max(t_exp, 1.0)

    # Determine limiting factor
    shot_noise_sq = source_flux * t_exp
    sky_noise_sq = sky_background_e_per_s * t_exp
    read_noise_sq = read_noise_e ** 2
    total_noise_sq = shot_noise_sq + sky_noise_sq + read_noise_sq
    if read_noise_sq > 0.5 * total_noise_sq:
        limiting_factor = "readout"
    elif t_exp >= transit_sec * 0.9:
        limiting_factor = "duration"
    else:
        limiting_factor = "photon_noise"

    # Total observations: transit + 30 min baseline each side
    baseline_sec = 3600.0
    total_obs_sec = transit_sec + baseline_sec
    effective_t = t_exp * (1.0 + overhead_fraction)
    n_exposures = max(1, int(transit_sec / effective_t))
    total_time_hours = total_obs_sec / 3600.0

    return TelescopeTimeResult(
        target_snr=target_snr,
        depth_ppm=depth_ppm,
        star_vmag=star_vmag,
        telescope_diameter_cm=telescope_diameter_cm,
        exposure_time_sec=round(t_exp, 2),
        n_exposures_in_transit=n_exposures,
        overhead_fraction=overhead_fraction,
        total_time_hours=round(total_time_hours, 3),
        is_feasible=total_time_hours <= max_hours,
        limiting_factor=limiting_factor,
        flag="OK",
    )


def format_telescope_time_result(result: TelescopeTimeResult) -> str:
    """Format telescope time estimate as Markdown."""
    lines = [
        "## Telescope Time Estimate",
        "",
        f"- Depth: {result.depth_ppm:.1f} ppm",
        f"- V-mag: {result.star_vmag:.2f}",
        f"- Telescope: {result.telescope_diameter_cm:.0f} cm",
        f"- Target SNR: {result.target_snr:.1f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID**")
    elif result.flag == "TOO_FAINT":
        lines.append("- **Flag: TOO_FAINT** — insufficient photon flux")
    else:
        lines += [
            f"- Exposure time: {result.exposure_time_sec:.1f} s",
            f"- Exposures in transit: {result.n_exposures_in_transit}",
            f"- Total time: {result.total_time_hours:.2f} h",
            f"- Feasible: {'Yes' if result.is_feasible else 'No'}",
            f"- Limiting factor: {result.limiting_factor}",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="telescope_time_estimator",
        description="Estimate telescope time for transit follow-up.",
    )
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("star_vmag", type=float)
    parser.add_argument("telescope_diameter_cm", type=float)
    parser.add_argument("--target-snr", type=float, default=10.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args(argv)

    result = estimate_telescope_time(
        args.depth_ppm, args.duration_hours, args.star_vmag, args.telescope_diameter_cm,
        target_snr=args.target_snr, max_hours=args.max_hours,
    )
    print(format_telescope_time_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
