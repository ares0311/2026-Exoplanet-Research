"""Feature extraction: raw diagnostic inputs → normalized [0, 1] feature scores."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from exo_toolkit.schemas import CandidateFeatures, CandidateSignal

# One solar radius expressed in AU (used for transit duration plausibility).
_R_SUN_AU: float = 0.00465047


# ---------------------------------------------------------------------------
# Raw diagnostics container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDiagnostics:
    """
    Raw measured quantities collected before feature extraction.

    All fields are optional; None propagates to a None feature score, which
    the scoring engine treats as "diagnostic not available."  Sequences are
    typed as tuples to match the frozen-dataclass semantics.
    """

    # One measurement per observed transit
    individual_depths: tuple[float, ...] | None = None
    individual_depth_errors: tuple[float, ...] | None = None
    individual_durations: tuple[float, ...] | None = None
    individual_duration_errors: tuple[float, ...] | None = None
    individual_transit_midpoints: tuple[float, ...] | None = None  # BJD

    # Per-sector measurements
    sector_depths: tuple[float, ...] | None = None        # per-sector mean transit depths (ppm)
    sector_depth_errors: tuple[float, ...] | None = None  # per-sector depth uncertainties (ppm)

    # Odd/even transit depth comparison
    depth_odd_ppm: float | None = None
    err_odd_ppm: float | None = None
    depth_even_ppm: float | None = None
    err_even_ppm: float | None = None

    # Secondary eclipse search result
    secondary_snr: float | None = None

    # Transit morphology: 0.0 = pure V-shape, 1.0 = fully flat-bottomed box
    ingress_egress_fraction: float | None = None

    # Stellar parameters for orbit plausibility checks
    stellar_radius_rsun: float | None = None
    stellar_mass_msun: float | None = None
    stellar_teff_k: float | None = None

    # Aperture contamination and centroid diagnostics
    contamination_ratio: float | None = None         # flux fraction from other sources
    centroid_offset_sigma: float | None = None        # significance of in-transit centroid shift
    centroid_motion_arcsec: float | None = None       # in-transit centroid displacement (arcsec)

    # Background eclipsing binary indicators
    nearby_bright_source_count: int | None = None     # Gaia/TIC sources inside aperture
    nearby_source_magnitude_diff: float | None = None  # Δmag of nearest contaminating source
    aperture_edge_proximity: float | None = None      # 0 = centered, 1 = at aperture edge

    # Instrumental systematics
    quality_flag_fraction: float | None = None        # fraction of transits with TESS quality flags
    sector_boundary_fraction: float | None = None     # fraction of events near sector boundaries
    background_excursion_sigma: float | None = None   # peak background excursion during transit (σ)
    data_gap_fraction: float | None = None            # fraction of transits falling in data gaps
    nearby_targets_common_signal: float | None = None  # cross-target correlation coefficient
    oot_scatter_sigma: float | None = None  # out-of-transit scatter (units: expected photon noise)

    # Stellar variability indicators
    ls_power_at_period: float | None = None           # Lomb-Scargle power at transit period
    ls_power_at_harmonics: float | None = None        # LS power at harmonic/sub-harmonic periods
    flare_rate_per_day: float | None = None
    quasi_periodic_strength: float | None = None      # autocorrelation amplitude [0, 1]

    # Known-object catalog matching
    target_id_matched: bool | None = None
    period_match_sigma: float | None = None           # σ deviation from catalog period
    epoch_match_sigma: float | None = None            # σ deviation from catalog epoch
    coordinate_match_arcsec: float | None = None      # angular separation to nearest catalog match


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _robust_cv(values: tuple[float, ...]) -> float:
    """Median absolute deviation / |median| — robust coefficient of variation."""
    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    if abs(median) < 1e-10:
        return 1.0  # undefined; treat as maximally inconsistent
    mad = float(np.median(np.abs(arr - median)))
    return mad / abs(median)


# ---------------------------------------------------------------------------
# Detection quality features
# ---------------------------------------------------------------------------


def snr_score(snr: float) -> float:
    """Linear ramp from 0 at SNR 5 to 1 at SNR 12."""
    return _clip((snr - 5.0) / (12.0 - 5.0))


def log_snr_score(snr: float) -> float:
    """Log-compressed SNR; more stable than the linear form at high SNR."""
    return _clip(math.log(max(snr, 1.0)) / math.log(12.0))


def transit_count_score(n: int) -> float:
    """
    Credibility contribution from the number of observed transits.
    Single-transit events are weak; three or more are strong.
    """
    if n >= 3:
        return 1.00
    if n == 2:
        return 0.70
    return 0.25  # n == 1


def depth_consistency_score(
    depths: tuple[float, ...],
    errors: tuple[float, ...],  # reserved for error-weighted version
    cv_threshold: float = 0.30,
) -> float | None:
    """
    Measures whether individual transit depths agree across events.
    Returns None when fewer than two transits are available.

    cv_threshold: robust CV value at which the score reaches 0.
    """
    if len(depths) < 2:
        return None
    cv = _robust_cv(depths)
    return _clip(1.0 - cv / cv_threshold)


def duration_consistency_score(
    durations: tuple[float, ...],
    errors: tuple[float, ...],  # reserved for error-weighted version
    cv_threshold: float = 0.25,
) -> float | None:
    """
    Measures whether transit durations are stable across events.
    Returns None when fewer than two transits are available.
    """
    if len(durations) < 2:
        return None
    cv = _robust_cv(durations)
    return _clip(1.0 - cv / cv_threshold)


def transit_timing_variation_score(
    midpoints: tuple[float, ...],
    period_days: float,
    epoch_bjd: float,
    rms_threshold_minutes: float = 10.0,
) -> float | None:
    """
    Observed-minus-computed (O-C) scatter of transit midpoints.

    Measures how much the observed transit times deviate from a strict linear
    ephemeris (constant period).  High scatter relative to the threshold
    suggests instrumental artifacts or non-periodic events.

    Score = clip(rms_oc_minutes / rms_threshold_minutes).
    Returns None when fewer than two midpoints are available.
    """
    if len(midpoints) < 2:
        return None
    arr = np.array(midpoints, dtype=float)
    transit_numbers = np.round((arr - epoch_bjd) / period_days).astype(int)
    predicted = epoch_bjd + transit_numbers * period_days
    oc_minutes = (arr - predicted) * 1440.0  # days → minutes
    rms = float(np.sqrt(np.mean(oc_minutes**2)))
    return _clip(rms / rms_threshold_minutes)


def out_of_transit_scatter_score(
    oot_scatter_sigma: float,
    sigma_threshold: float = 3.0,
) -> float:
    return _clip(oot_scatter_sigma / sigma_threshold)


def multi_sector_depth_consistency_score(
    sector_depths: tuple[float, ...],
    sector_depth_errors: tuple[float, ...] | None = None,
    cv_threshold: float = 0.20,
) -> float | None:
    if len(sector_depths) < 2:
        return None
    cv = _robust_cv(sector_depths)
    return _clip(1.0 - cv / cv_threshold)


_G_CGS = 6.674e-8          # cm³ g⁻¹ s⁻²
_RSUN_CM = 6.957e10        # cm
_MSUN_G = 1.989e33         # g
_SECONDS_PER_DAY = 86400.0


def stellar_density_consistency_score(
    duration_hours: float,
    period_days: float,
    depth_ppm: float,
    stellar_radius_rsun: float,
    stellar_mass_msun: float,
    tolerance_factor: float = 3.0,
) -> float:
    """Compare photometric stellar density to catalog density."""
    if period_days <= 0 or duration_hours <= 0 or depth_ppm <= 0:
        return 0.0
    R = stellar_radius_rsun * _RSUN_CM
    M = stellar_mass_msun * _MSUN_G
    rho_cat = M / (4.0 / 3.0 * 3.14159265 * R**3)
    T = duration_hours * 3600.0
    P = period_days * _SECONDS_PER_DAY
    aR = P / (3.14159265 * T)  # a/R_star from transit duration approximation (b=0)
    rho_phot = (3.0 * 3.14159265 / (_G_CGS * P**2)) * aR**3
    if rho_cat <= 0:
        return 0.0
    discrepancy = abs(rho_phot - rho_cat) / (tolerance_factor * rho_cat)
    return _clip(1.0 - discrepancy)


def centroid_motion_score(
    centroid_motion_arcsec: float,
    saturation_arcsec: float = 2.0,
) -> float:
    return _clip(centroid_motion_arcsec / saturation_arcsec)


def _ld_ingress_egress_fraction(depth_ppm: float, stellar_teff_k: float) -> float:
    """Expected ingress+egress fraction via quadratic LD approximation."""
    teff_norm = max(0.0, min(1.0, (stellar_teff_k - 3000.0) / 7000.0))
    u_sum = 0.30 + 0.50 * (1.0 - teff_norm)
    k = (depth_ppm / 1e6) ** 0.5
    return _clip(2.0 * k / (1.0 + u_sum / 6.0))


def limb_darkening_plausibility_score(
    ingress_egress_fraction: float,
    depth_ppm: float,
    stellar_teff_k: float = 5778.0,
) -> float:
    expected_ief = _ld_ingress_egress_fraction(depth_ppm, stellar_teff_k)
    if expected_ief <= 0:
        return 0.5
    discrepancy = abs(ingress_egress_fraction - expected_ief) / max(expected_ief, 0.01)
    return _clip(1.0 - discrepancy)


def duration_plausibility_score(
    duration_hours: float,
    period_days: float,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
) -> float:
    """
    Score based on whether the transit duration is consistent with a bound
    planetary orbit around the given star.

    Computes T_max for a central (b=0), circular-orbit transit using
    Kepler's third law and the small-planet approximation:

        a  = (M_star * P²)^(1/3)   [solar units, AU]
        T_max = (P / π) * arcsin(R_star / a)   [hours]

    Score is 1 inside the plausible range [0.1·T_max, 1.5·T_max], decays
    toward 0 for durations that are too short (grazing/unresolved) or too
    long (stellar companion).
    """
    a_au = (stellar_mass_msun * (period_days / 365.25) ** 2) ** (1.0 / 3.0)
    r_star_au = stellar_radius_rsun * _R_SUN_AU
    sin_arg = _clip(r_star_au / a_au, -1.0, 1.0)
    t_max_hours = (period_days * 24.0 / math.pi) * math.asin(sin_arg)

    t_min_hours = t_max_hours * 0.10   # near-grazing transit
    t_upper_hours = t_max_hours * 1.50  # generous margin for orbit uncertainty

    if duration_hours <= 0.0:
        return 0.0

    if t_min_hours <= duration_hours <= t_upper_hours:
        overshoot = max(0.0, duration_hours - t_max_hours) / max(
            t_upper_hours - t_max_hours, 1e-10
        )
        return _clip(1.0 - 0.5 * overshoot)

    if duration_hours > t_upper_hours:
        # Implausibly long — likely a stellar companion
        return _clip(t_upper_hours / duration_hours)

    # Too short — below grazing-transit minimum
    return _clip(duration_hours / max(t_min_hours, 1e-10))


# ---------------------------------------------------------------------------
# Eclipsing binary indicators
# ---------------------------------------------------------------------------


def odd_even_mismatch_score(
    depth_odd: float,
    err_odd: float,
    depth_even: float,
    err_even: float,
) -> float:
    """
    Significance of the depth difference between odd and even transits,
    normalised so that a 5-sigma mismatch saturates the score at 1.

    High score → likely on-target eclipsing binary.
    """
    combined_err = math.sqrt(err_odd**2 + err_even**2)
    if combined_err < 1e-10:
        return 0.0
    sigma = abs(depth_odd - depth_even) / combined_err
    return _clip(sigma / 5.0)


def secondary_eclipse_score(secondary_snr: float) -> float:
    """
    Likelihood of a secondary eclipse near phase 0.5.
    SNR of 7 saturates the score.

    High score → likely eclipsing binary or self-luminous companion.
    """
    return _clip(secondary_snr / 7.0)


def transit_shape_score(ingress_egress_fraction: float) -> float:
    """
    Measures how box/U-shaped (flat-bottomed) the transit is.
    0 = pure V-shape, 1 = fully flat-bottomed.

    High score → supports planet candidate hypothesis.
    """
    return _clip(ingress_egress_fraction)


def v_shape_score(ingress_egress_fraction: float) -> float:
    """
    Inverse of transit_shape_score.
    High score → supports eclipsing binary hypothesis.
    """
    return _clip(1.0 - ingress_egress_fraction)


def non_box_shape_score(ingress_egress_fraction: float) -> float:
    """
    Measures how far the event departs from an idealised box transit.
    Complements v_shape_score for the stellar variability hypothesis.
    """
    return _clip(1.0 - ingress_egress_fraction)


def large_depth_score(depth_ppm: float) -> float:
    """
    Penalises very deep transits more consistent with a stellar eclipse.
    Score rises from 0 at 10 000 ppm (1 %) to 1 at 100 000 ppm (10 %).

    Planets rarely exceed ~1 % depth; eclipsing binaries commonly do.
    """
    return _clip((depth_ppm - 10_000.0) / 90_000.0)


def companion_radius_too_large_score(
    depth_ppm: float,
    stellar_radius_rsun: float = 1.0,
) -> float:
    """
    Infers companion radius from transit depth and scores how unlikely
    that radius is for a planet.

    R_companion / R_star = sqrt(depth_ppm / 1e6), so
    R_companion [R_Sun] = stellar_radius_rsun * sqrt(depth_ppm / 1e6).

    Score rises from 0 at 0.15 R_Sun (~1.5 R_Jup) to 1 at 0.50 R_Sun.
    """
    r_companion = stellar_radius_rsun * math.sqrt(max(depth_ppm, 0.0) / 1e6)
    return _clip((r_companion - 0.15) / (0.50 - 0.15))


def duration_implausibility_score(
    duration_hours: float,
    period_days: float,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
) -> float:
    """
    Inverse of duration_plausibility_score.
    High score → duration is inconsistent with a planetary orbit.
    """
    return _clip(
        1.0 - duration_plausibility_score(
            duration_hours, period_days, stellar_radius_rsun, stellar_mass_msun
        )
    )


# ---------------------------------------------------------------------------
# Background eclipsing binary indicators
# ---------------------------------------------------------------------------


def contamination_score(contamination_ratio: float) -> float:
    """
    Fraction of aperture flux from sources other than the target.
    Higher contamination → higher false-positive risk.
    """
    return _clip(contamination_ratio)


def centroid_offset_score(centroid_offset_sigma: float) -> float:
    """
    Significance of the flux centroid shift during transit.
    A 5-sigma offset saturates the score.

    High score → flux source is off-target → supports background EB.
    """
    return _clip(centroid_offset_sigma / 5.0)


def nearby_bright_source_score(
    count: int,
    magnitude_diff: float | None,
) -> float:
    """
    Blending risk from nearby bright sources inside the aperture.
    Combines source count (saturates at 3) and magnitude difference
    (brighter neighbours are more dangerous, i.e. smaller Δmag).
    """
    count_component = _clip(count / 3.0)
    if magnitude_diff is None:
        return count_component
    mag_component = _clip(1.0 - magnitude_diff / 5.0)
    return _clip(0.5 * count_component + 0.5 * mag_component)


def aperture_edge_score(proximity: float) -> float:
    """Score based on how close the centroid is to the aperture edge."""
    return _clip(proximity)


def dilution_sensitivity_score(contamination_ratio: float) -> float:
    """
    How unreliable the measured depth is due to third-light dilution.
    Uses the same contamination ratio as contamination_score but
    captures the background-EB-specific concern about depth correction.
    """
    return _clip(contamination_ratio)


# ---------------------------------------------------------------------------
# Stellar variability indicators
# ---------------------------------------------------------------------------


def variability_periodogram_score(ls_power: float) -> float:
    """
    Lomb-Scargle power at or near the transit period.
    Power ≥ 0.5 is treated as strong evidence for stellar variability.
    """
    return _clip(ls_power / 0.5)


def harmonic_score(ls_power_at_harmonics: float) -> float:
    """
    LS power at integer harmonics or sub-harmonics of the transit period.
    High power → rotational modulation rather than a planet.
    """
    return _clip(ls_power_at_harmonics / 0.5)


def flare_score(flare_rate_per_day: float) -> float:
    """
    Activity indicator from observed flare rate.
    Saturates at 2 flares / day.
    """
    return _clip(flare_rate_per_day / 2.0)


def quasi_periodic_score(quasi_periodic_strength: float) -> float:
    """
    Autocorrelation amplitude measuring quasi-periodic variability.
    Input expected in [0, 1]: 0 = white noise, 1 = perfectly periodic.
    """
    return _clip(quasi_periodic_strength)


def stellar_variability_score(
    ls_power: float | None,
    ls_harmonics: float | None,
    flare_rate: float | None,
    quasi_periodic: float | None,
) -> float | None:
    """
    Combined stellar variability indicator (mean of available components).
    Returns None only when all inputs are None.
    """
    components: list[float] = []
    if ls_power is not None:
        components.append(variability_periodogram_score(ls_power))
    if ls_harmonics is not None:
        components.append(harmonic_score(ls_harmonics))
    if flare_rate is not None:
        components.append(flare_score(flare_rate))
    if quasi_periodic is not None:
        components.append(quasi_periodic_score(quasi_periodic))
    if not components:
        return None
    return _clip(sum(components) / len(components))


# ---------------------------------------------------------------------------
# Instrumental artifact indicators
# ---------------------------------------------------------------------------


def quality_flag_score(quality_flag_fraction: float) -> float:
    """Fraction of transits coinciding with TESS/Kepler quality flags."""
    return _clip(quality_flag_fraction)


def sector_boundary_score(sector_boundary_fraction: float) -> float:
    """Fraction of events falling near TESS sector boundaries."""
    return _clip(sector_boundary_fraction)


def background_excursion_score(background_excursion_sigma: float) -> float:
    """
    Peak background flux excursion during transit, in sigma.
    A 5-sigma excursion saturates the score.
    """
    return _clip(background_excursion_sigma / 5.0)


def single_event_score(transit_count: int) -> float:
    """1.0 for single-transit events (poor repeatability); 0.0 otherwise."""
    return 1.0 if transit_count == 1 else 0.0


def nearby_targets_common_signal_score(correlation: float) -> float:
    """
    Cross-correlation with neighbouring target light curves.
    High correlation → common-mode systematic, not an astrophysical signal.
    """
    return _clip(correlation)


def data_gap_overlap_score(data_gap_fraction: float) -> float:
    """Fraction of expected transit midpoints that fall inside data gaps."""
    return _clip(data_gap_fraction)


def depth_scatter_chi2_score(
    depths: tuple[float, ...],
    errors: tuple[float, ...],
    chi2_threshold: float = 3.0,
) -> float | None:
    """
    Reduced chi-square of individual transit depths relative to measurement errors.
    Returns None when fewer than two transits are available or any error is ≤ 0.

    High score → depths vary more than expected from noise alone, suggesting
    instrumental artifacts or a non-periodic astrophysical contaminant.
    """
    if len(depths) < 2 or len(errors) < 2:
        return None
    depths_arr = np.array(depths, dtype=float)
    errors_arr = np.array(errors, dtype=float)
    if np.any(errors_arr <= 0.0):
        return None
    weights = 1.0 / errors_arr**2
    weighted_mean = float(np.dot(weights, depths_arr) / weights.sum())
    chi2_reduced = float(
        np.sum(((depths_arr - weighted_mean) / errors_arr) ** 2) / (len(depths) - 1)
    )
    return _clip(chi2_reduced / chi2_threshold)


def systematics_overlap_score(
    quality_flag_fraction: float | None,
    sector_boundary_fraction: float | None,
    background_excursion_sigma: float | None,
) -> float | None:
    """
    Combined instrumental systematics indicator.
    Takes the maximum of available components — any single serious systematic
    is sufficient to flag the event.  Returns None when all inputs are None.
    """
    components: list[float] = []
    if quality_flag_fraction is not None:
        components.append(quality_flag_score(quality_flag_fraction))
    if sector_boundary_fraction is not None:
        components.append(sector_boundary_score(sector_boundary_fraction))
    if background_excursion_sigma is not None:
        components.append(background_excursion_score(background_excursion_sigma))
    if not components:
        return None
    return _clip(max(components))


# ---------------------------------------------------------------------------
# Known-object indicators
# ---------------------------------------------------------------------------


def target_id_match_score(matched: bool) -> float:
    """Binary: did the target ID match a known-object catalog entry?"""
    return 1.0 if matched else 0.0


def period_match_score(match_sigma: float) -> float:
    """
    Agreement between candidate period and a catalog period.
    0 sigma → 1.0; 3 sigma → 0.0.
    """
    return _clip(1.0 - match_sigma / 3.0)


def epoch_match_score(match_sigma: float) -> float:
    """Same structure as period_match_score but for epoch."""
    return _clip(1.0 - match_sigma / 3.0)


def coordinate_match_score(separation_arcsec: float) -> float:
    """
    Angular proximity to a catalog object.
    0 arcsec → 1.0; 30 arcsec → 0.0.
    """
    return _clip(1.0 - separation_arcsec / 30.0)


def known_object_score(
    target_id_matched: bool | None,
    period_match_sigma: float | None,
    epoch_match_sigma: float | None,
    coordinate_match_arcsec: float | None,
) -> float | None:
    """
    Weighted combination of catalog-matching sub-scores.
    Weights follow SCORING_MODEL.md §7 (known object log-score weights).
    Returns None when all inputs are None.
    """
    components: list[tuple[float, float]] = []  # (score, weight)
    if target_id_matched is not None:
        components.append((target_id_match_score(target_id_matched), 2.5))
    if period_match_sigma is not None:
        components.append((period_match_score(period_match_sigma), 2.0))
    if epoch_match_sigma is not None:
        components.append((epoch_match_score(epoch_match_sigma), 1.5))
    if coordinate_match_arcsec is not None:
        components.append((coordinate_match_score(coordinate_match_arcsec), 1.2))
    if not components:
        return None
    total_weight = sum(w for _, w in components)
    return _clip(sum(s * w for s, w in components) / total_weight)


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------


def extract_features(
    signal: CandidateSignal,
    diagnostics: RawDiagnostics,
) -> CandidateFeatures:
    """
    Compute all available feature scores from a signal and its raw diagnostics.
    Features whose required diagnostics are absent are set to None.
    """
    d = diagnostics
    r_rsun = d.stellar_radius_rsun if d.stellar_radius_rsun is not None else 1.0
    m_msun = d.stellar_mass_msun if d.stellar_mass_msun is not None else 1.0

    # --- detection quality (always computable from signal) ---
    snr_s = snr_score(signal.snr)
    log_snr_s = log_snr_score(signal.snr)
    tc_s = transit_count_score(signal.transit_count)
    dur_plaus_s = duration_plausibility_score(
        signal.duration_hours, signal.period_days, r_rsun, m_msun
    )

    depth_cons_s: float | None = None
    depth_chi2_s: float | None = None
    if d.individual_depths is not None and d.individual_depth_errors is not None:
        depth_cons_s = depth_consistency_score(
            d.individual_depths, d.individual_depth_errors
        )
        depth_chi2_s = depth_scatter_chi2_score(
            d.individual_depths, d.individual_depth_errors
        )

    dur_cons_s: float | None = None
    if d.individual_durations is not None and d.individual_duration_errors is not None:
        dur_cons_s = duration_consistency_score(
            d.individual_durations, d.individual_duration_errors
        )

    ttv_s: float | None = None
    if d.individual_transit_midpoints is not None:
        ttv_s = transit_timing_variation_score(
            d.individual_transit_midpoints, signal.period_days, signal.epoch_bjd
        )

    # --- transit morphology ---
    shape_s: float | None = None
    v_s: float | None = None
    non_box_s: float | None = None
    if d.ingress_egress_fraction is not None:
        shape_s = transit_shape_score(d.ingress_egress_fraction)
        v_s = v_shape_score(d.ingress_egress_fraction)
        non_box_s = non_box_shape_score(d.ingress_egress_fraction)

    # --- eclipsing binary (always computable from signal + optional stellar params) ---
    large_d_s = large_depth_score(signal.depth_ppm)
    comp_r_s = companion_radius_too_large_score(signal.depth_ppm, r_rsun)
    dur_implaus_s = duration_implausibility_score(
        signal.duration_hours, signal.period_days, r_rsun, m_msun
    )

    odd_even_s: float | None = None
    if (
        d.depth_odd_ppm is not None
        and d.err_odd_ppm is not None
        and d.depth_even_ppm is not None
        and d.err_even_ppm is not None
    ):
        odd_even_s = odd_even_mismatch_score(
            d.depth_odd_ppm, d.err_odd_ppm, d.depth_even_ppm, d.err_even_ppm
        )

    sec_s: float | None = None
    if d.secondary_snr is not None:
        sec_s = secondary_eclipse_score(d.secondary_snr)

    # --- background eclipsing binary ---
    cont_s: float | None = None
    dil_s: float | None = None
    if d.contamination_ratio is not None:
        cont_s = contamination_score(d.contamination_ratio)
        dil_s = dilution_sensitivity_score(d.contamination_ratio)

    cent_s: float | None = None
    if d.centroid_offset_sigma is not None:
        cent_s = centroid_offset_score(d.centroid_offset_sigma)

    nearby_s: float | None = None
    if d.nearby_bright_source_count is not None:
        nearby_s = nearby_bright_source_score(
            d.nearby_bright_source_count, d.nearby_source_magnitude_diff
        )

    aper_s: float | None = None
    if d.aperture_edge_proximity is not None:
        aper_s = aperture_edge_score(d.aperture_edge_proximity)

    # --- stellar variability ---
    var_s: float | None = None
    if d.ls_power_at_period is not None:
        var_s = variability_periodogram_score(d.ls_power_at_period)

    harm_s: float | None = None
    if d.ls_power_at_harmonics is not None:
        harm_s = harmonic_score(d.ls_power_at_harmonics)

    flare_s: float | None = None
    if d.flare_rate_per_day is not None:
        flare_s = flare_score(d.flare_rate_per_day)

    qp_s: float | None = None
    if d.quasi_periodic_strength is not None:
        qp_s = quasi_periodic_score(d.quasi_periodic_strength)

    sv_s = stellar_variability_score(
        d.ls_power_at_period,
        d.ls_power_at_harmonics,
        d.flare_rate_per_day,
        d.quasi_periodic_strength,
    )

    # --- instrumental systematics ---
    sys_s = systematics_overlap_score(
        d.quality_flag_fraction,
        d.sector_boundary_fraction,
        d.background_excursion_sigma,
    )

    qf_s: float | None = None
    if d.quality_flag_fraction is not None:
        qf_s = quality_flag_score(d.quality_flag_fraction)

    sb_s: float | None = None
    if d.sector_boundary_fraction is not None:
        sb_s = sector_boundary_score(d.sector_boundary_fraction)

    be_s: float | None = None
    if d.background_excursion_sigma is not None:
        be_s = background_excursion_score(d.background_excursion_sigma)

    se_s = single_event_score(signal.transit_count)

    ntcs: float | None = None
    if d.nearby_targets_common_signal is not None:
        ntcs = nearby_targets_common_signal_score(d.nearby_targets_common_signal)

    dg_s: float | None = None
    if d.data_gap_fraction is not None:
        dg_s = data_gap_overlap_score(d.data_gap_fraction)

    oot_s: float | None = None
    if d.oot_scatter_sigma is not None:
        oot_s = out_of_transit_scatter_score(d.oot_scatter_sigma)

    ms_depth_s: float | None = None
    if d.sector_depths is not None:
        ms_depth_s = multi_sector_depth_consistency_score(d.sector_depths, d.sector_depth_errors)

    density_s: float | None = None
    if d.stellar_radius_rsun is not None and d.stellar_mass_msun is not None:
        density_s = stellar_density_consistency_score(
            signal.duration_hours,
            signal.period_days,
            signal.depth_ppm,
            d.stellar_radius_rsun,
            d.stellar_mass_msun,
        )

    cm_motion_s: float | None = None
    if d.centroid_motion_arcsec is not None:
        cm_motion_s = centroid_motion_score(d.centroid_motion_arcsec)

    ld_s: float | None = None
    if d.ingress_egress_fraction is not None:
        teff = d.stellar_teff_k if d.stellar_teff_k is not None else 5778.0
        ld_s = limb_darkening_plausibility_score(
            d.ingress_egress_fraction, signal.depth_ppm, teff
        )

    # --- known object ---
    ko_s = known_object_score(
        d.target_id_matched,
        d.period_match_sigma,
        d.epoch_match_sigma,
        d.coordinate_match_arcsec,
    )

    tid_s: float | None = None
    if d.target_id_matched is not None:
        tid_s = target_id_match_score(d.target_id_matched)

    pm_s: float | None = None
    if d.period_match_sigma is not None:
        pm_s = period_match_score(d.period_match_sigma)

    em_s: float | None = None
    if d.epoch_match_sigma is not None:
        em_s = epoch_match_score(d.epoch_match_sigma)

    cm_s: float | None = None
    if d.coordinate_match_arcsec is not None:
        cm_s = coordinate_match_score(d.coordinate_match_arcsec)

    return CandidateFeatures(
        # detection quality
        snr_score=snr_s,
        log_snr_score=log_snr_s,
        transit_count_score=tc_s,
        depth_consistency_score=depth_cons_s,
        duration_consistency_score=dur_cons_s,
        duration_plausibility_score=dur_plaus_s,
        transit_shape_score=shape_s,
        data_gap_overlap_score=dg_s,
        transit_timing_variation_score=ttv_s,
        out_of_transit_scatter_score=oot_s,
        multi_sector_depth_consistency_score=ms_depth_s,
        stellar_density_consistency_score=density_s,
        limb_darkening_plausibility_score=ld_s,
        # eclipsing binary
        odd_even_mismatch_score=odd_even_s,
        secondary_eclipse_score=sec_s,
        v_shape_score=v_s,
        large_depth_score=large_d_s,
        companion_radius_too_large_score=comp_r_s,
        duration_implausibility_score=dur_implaus_s,
        # background eclipsing binary
        centroid_offset_score=cent_s,
        contamination_score=cont_s,
        nearby_bright_source_score=nearby_s,
        aperture_edge_score=aper_s,
        dilution_sensitivity_score=dil_s,
        centroid_motion_score=cm_motion_s,
        # stellar variability
        stellar_variability_score=sv_s,
        variability_periodogram_score=var_s,
        harmonic_score=harm_s,
        flare_score=flare_s,
        quasi_periodic_score=qp_s,
        non_box_shape_score=non_box_s,
        # instrumental
        systematics_overlap_score=sys_s,
        quality_flag_score=qf_s,
        sector_boundary_score=sb_s,
        background_excursion_score=be_s,
        single_event_score=se_s,
        nearby_targets_common_signal_score=ntcs,
        depth_scatter_chi2_score=depth_chi2_s,
        # known object
        known_object_score=ko_s,
        target_id_match_score=tid_s,
        period_match_score=pm_s,
        epoch_match_score=em_s,
        coordinate_match_score=cm_s,
    )
