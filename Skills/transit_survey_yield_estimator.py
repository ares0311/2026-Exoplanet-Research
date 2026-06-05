"""Estimate expected planet detections from a transit survey."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyYieldResult:
    expected_detections: float
    geometric_efficiency: float     # mean geometric transit probability
    window_efficiency: float        # fraction of orbital period covered
    combined_efficiency: float      # geometric × window
    detections_per_period_bin: tuple[float, ...]
    flag: str


_PERIOD_BINS_DAYS = (1.0, 3.0, 10.0, 30.0, 100.0, 365.0)


def estimate_survey_yield(
    n_stars: int,
    observation_baseline_days: float,
    eta_earth: float = 0.1,
    cdpp_ppm_per_hour: float = 100.0,
    snr_threshold: float = 7.5,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
    planet_radius_rearth: float = 2.0,
) -> SurveyYieldResult:
    """Estimate expected number of transit detections from a survey.

    For each period bin, compute:
      p_geom(P) = Rs / a(P)
      p_window(P) = min(baseline/P, 1) * min(n_transits * T_tr / baseline, 1)
      yield(P) += N_stars × η_Earth × p_geom × p_window × p_photo

    Args:
        n_stars: number of target stars
        observation_baseline_days: total observation baseline (days)
        eta_earth: occurrence rate per star (Earth-like planets)
        cdpp_ppm_per_hour: photometric precision (ppm/√hr)
        snr_threshold: minimum detection SNR
        stellar_radius_rsun: typical stellar radius (solar radii)
        stellar_mass_msun: typical stellar mass (solar masses)
        planet_radius_rearth: typical planet radius (Earth radii)
    """
    _G = 6.674e-11
    _MSUN_KG = 1.989e30
    _RSUN_M = 6.957e8
    _REARTH_M = 6.371e6

    if n_stars <= 0:
        return SurveyYieldResult(float("nan"), float("nan"), float("nan"),
                                  float("nan"), (), "INVALID_N_STARS")
    if observation_baseline_days <= 0.0:
        return SurveyYieldResult(float("nan"), float("nan"), float("nan"),
                                  float("nan"), (), "INVALID_BASELINE")
    if not (0.0 < eta_earth <= 1.0):
        return SurveyYieldResult(float("nan"), float("nan"), float("nan"),
                                  float("nan"), (), "INVALID_ETA")

    ms_kg = stellar_mass_msun * _MSUN_KG
    rs_m = stellar_radius_rsun * _RSUN_M
    rp_m = planet_radius_rearth * _REARTH_M

    total_yield = 0.0
    bin_yields: list[float] = []
    sum_geom = 0.0
    sum_window = 0.0

    for period_days in _PERIOD_BINS_DAYS:
        p_s = period_days * 86400.0
        a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)

        p_geom = min((rs_m + rp_m) / a_m, 1.0)

        n_transits = observation_baseline_days / period_days
        t_tr_hr = max((period_days / math.pi) * math.asin(min(rs_m / a_m, 1.0)) * 24.0,
                      0.5)

        depth_ppm = (rp_m / rs_m) ** 2 * 1e6
        snr = depth_ppm * math.sqrt(n_transits * t_tr_hr) / max(cdpp_ppm_per_hour, 1.0)
        p_photo = 1.0 if snr >= snr_threshold else 0.0

        p_window = min(n_transits, 1.0)

        y = n_stars * eta_earth * p_geom * p_window * p_photo
        bin_yields.append(round(y, 4))
        total_yield += y
        sum_geom += p_geom
        sum_window += p_window

    n_bins = len(_PERIOD_BINS_DAYS)
    mean_geom = sum_geom / n_bins
    mean_window = sum_window / n_bins

    return SurveyYieldResult(
        expected_detections=total_yield,
        geometric_efficiency=mean_geom,
        window_efficiency=mean_window,
        combined_efficiency=mean_geom * mean_window,
        detections_per_period_bin=tuple(bin_yields),
        flag="OK",
    )


def format_survey_yield_result(r: SurveyYieldResult) -> str:
    if r.flag != "OK":
        return f"SurveyYield | flag={r.flag}"
    rows = "\n".join(
        f"| {p:.0f} d | {y:.4f} |"
        for p, y in zip(_PERIOD_BINS_DAYS, r.detections_per_period_bin, strict=False)
    )
    return (
        f"Expected detections: {r.expected_detections:.2f} | "
        f"Geometric eff.: {r.geometric_efficiency:.4f} | "
        f"flag={r.flag}\n\n"
        f"| Period bin | Expected yield |\n"
        f"|---|---|\n"
        f"{rows}"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit survey yield estimator")
    p.add_argument("n_stars", type=int)
    p.add_argument("baseline_days", type=float)
    p.add_argument("--eta", type=float, default=0.1)
    args = p.parse_args()
    r = estimate_survey_yield(args.n_stars, args.baseline_days, eta_earth=args.eta)
    print(format_survey_yield_result(r))


if __name__ == "__main__":
    _cli()
