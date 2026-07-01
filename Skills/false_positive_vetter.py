"""Structured false-positive vetting report for a pipeline candidate.

Maps each CandidateFeatures score to a pass/warn/fail verdict using
configurable thresholds.  Produces a Markdown checklist for manual review.

Public API
----------
vet_candidate(row, *, warn_threshold, fail_threshold) -> list[VetVerdict]
format_vetting_report(verdicts, candidate_id) -> str
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Thresholds per feature: (warn, fail)
# For FP-indicator scores, higher = more suspicious.
# For quality scores, lower = more suspicious (invert below).
# ---------------------------------------------------------------------------

_FP_INDICATOR_FEATURES = {
    "odd_even_mismatch_score":      (0.20, 0.50),
    "secondary_eclipse_score":      (0.15, 0.40),
    "centroid_offset_score":        (0.20, 0.50),
    "contamination_score":          (0.20, 0.50),
    "systematics_overlap_score":    (0.20, 0.50),
    "stellar_variability_score":    (0.25, 0.55),
    "depth_scatter_chi2_score":     (0.30, 0.60),
    "transit_timing_variation_score": (0.25, 0.55),
    "out_of_transit_scatter_score": (0.25, 0.55),
    "centroid_motion_score":        (0.20, 0.50),
    "v_shape_score":                (0.20, 0.50),
    "large_depth_score":            (0.25, 0.55),
    "companion_radius_too_large_score": (0.20, 0.50),
    "single_event_score":           (0.30, 0.60),
    "quality_flag_score":           (0.20, 0.50),
    "sector_boundary_score":        (0.20, 0.50),
    "background_excursion_score":   (0.20, 0.50),
    "nearby_bright_source_score":   (0.20, 0.50),
    "aperture_edge_score":          (0.20, 0.50),
    "dilution_sensitivity_score":   (0.25, 0.55),
    "nearby_targets_common_signal_score": (0.25, 0.55),
    "non_box_shape_score":          (0.25, 0.55),
    "duration_implausibility_score": (0.25, 0.55),
}

# Quality scores: lower is more suspicious (we check 1 - score < threshold)
_QUALITY_FEATURES = {
    "log_snr_score":                (0.30, 0.15),
    "transit_count_score":          (0.30, 0.15),
    "depth_consistency_score":      (0.25, 0.10),
    "duration_consistency_score":   (0.25, 0.10),
    "duration_plausibility_score":  (0.25, 0.10),
    "transit_shape_score":          (0.25, 0.10),
    "multi_sector_depth_consistency_score": (0.25, 0.10),
    "stellar_density_consistency_score":    (0.25, 0.10),
    "limb_darkening_plausibility_score":    (0.25, 0.10),
}


@dataclass
class VetVerdict:
    feature_name: str
    score: float | None
    verdict: str      # "pass", "warn", "fail", "missing"
    threshold_warn: float
    threshold_fail: float
    is_fp_indicator: bool
    missing_reason: str | None = None


_MISSING_REASON_BY_FEATURE = {
    "odd_even_mismatch_score": (
        "needs at least four measured transit depths for an odd/even comparison"
    ),
    "secondary_eclipse_score": (
        "needs enough phase-0.5 coverage and out-of-eclipse baseline to measure a "
        "secondary eclipse"
    ),
    "duration_consistency_score": "needs per-transit duration fitting",
    "transit_timing_variation_score": "needs per-transit midpoint fitting",
    "out_of_transit_scatter_score": "needs expected photon-noise or OOT scatter diagnostics",
    "multi_sector_depth_consistency_score": "needs per-sector transit depth measurements",
    "stellar_density_consistency_score": "needs host-star radius and mass diagnostics",
    "centroid_offset_score": "needs in-transit centroid shift diagnostics",
    "centroid_motion_score": "needs in-transit centroid motion diagnostics",
    "contamination_score": "needs aperture contamination or CROWDSAP-style diagnostics",
    "dilution_sensitivity_score": "needs aperture contamination or CROWDSAP-style diagnostics",
    "nearby_bright_source_score": "needs nearby Gaia/TIC source diagnostics",
    "aperture_edge_score": "needs aperture-edge proximity diagnostics",
    "stellar_variability_score": "needs periodogram, flare, or quasi-periodic diagnostics",
    "variability_periodogram_score": "needs Lomb-Scargle power at the candidate period",
    "harmonic_score": "needs Lomb-Scargle power at harmonic/sub-harmonic periods",
    "flare_score": "needs flare-rate diagnostics",
    "quasi_periodic_score": "needs quasi-periodic variability diagnostics",
    "systematics_overlap_score": "needs quality-flag, sector-boundary, or background diagnostics",
    "quality_flag_score": "needs per-transit pipeline quality flag diagnostics",
    "sector_boundary_score": "needs sector-boundary overlap diagnostics",
    "background_excursion_score": "needs background-flux excursion diagnostics",
    "nearby_targets_common_signal_score": "needs neighboring-target common-signal diagnostics",
    "known_object_score": "needs known-object catalog crossmatch diagnostics",
    "target_id_match_score": "needs known-object target-ID crossmatch diagnostics",
    "period_match_score": "needs known-object period-match diagnostics",
    "epoch_match_score": "needs known-object epoch-match diagnostics",
    "coordinate_match_score": "needs known-object coordinate-match diagnostics",
}


def _missing_reason(feature_name: str, row: dict[str, Any]) -> str | None:
    diagnostics = row.get("diagnostics", {}) or {}

    if feature_name == "odd_even_mismatch_score":
        depths = diagnostics.get("individual_depths")
        n_depths = len(depths) if isinstance(depths, list | tuple) else 0
        if n_depths < 4:
            return (
                f"only {n_depths} measured transit depth(s); odd/even needs at least 4"
            )

    if feature_name == "multi_sector_depth_consistency_score":
        sector_depths = diagnostics.get("sector_depths")
        n_sectors = len(sector_depths) if isinstance(sector_depths, list | tuple) else 0
        if n_sectors < 2:
            return (
                f"only {n_sectors} sector depth measurement(s); multi-sector check needs at least 2"
            )

    if feature_name == "secondary_eclipse_score" and "secondary_snr" in diagnostics:
        return "secondary eclipse SNR was not measurable from the current phase coverage"

    return _MISSING_REASON_BY_FEATURE.get(feature_name)


def vet_candidate(
    row: dict[str, Any],
    *,
    warn_threshold_scale: float = 1.0,
    fail_threshold_scale: float = 1.0,
) -> list[VetVerdict]:
    """Evaluate a pipeline output row against FP thresholds.

    Args:
        row: Output row from ``run_pipeline()`` containing a ``"scores"`` key
            and optionally a ``"features"`` key with per-score values.
        warn_threshold_scale: Multiply all warn thresholds by this factor.
        fail_threshold_scale: Multiply all fail thresholds by this factor.

    Returns:
        List of :class:`VetVerdict` objects, one per checked feature.
    """
    features: dict[str, Any] = row.get("features", {}) or {}
    verdicts: list[VetVerdict] = []

    for name, (w, f) in _FP_INDICATOR_FEATURES.items():
        score = features.get(name)
        tw = w * warn_threshold_scale
        tf = f * fail_threshold_scale
        missing_reason = None
        if score is None:
            verdict = "missing"
            missing_reason = _missing_reason(name, row)
        elif score >= tf:
            verdict = "fail"
        elif score >= tw:
            verdict = "warn"
        else:
            verdict = "pass"
        verdicts.append(VetVerdict(
            feature_name=name, score=score,
            verdict=verdict, threshold_warn=tw, threshold_fail=tf,
            is_fp_indicator=True, missing_reason=missing_reason,
        ))

    for name, (w, f) in _QUALITY_FEATURES.items():
        score = features.get(name)
        tw = w
        tf = f
        missing_reason = None
        if score is None:
            verdict = "missing"
            missing_reason = _missing_reason(name, row)
        elif score < tf:
            verdict = "fail"
        elif score < tw:
            verdict = "warn"
        else:
            verdict = "pass"
        verdicts.append(VetVerdict(
            feature_name=name, score=score,
            verdict=verdict, threshold_warn=tw, threshold_fail=tf,
            is_fp_indicator=False, missing_reason=missing_reason,
        ))

    return verdicts


def format_vetting_report(
    verdicts: list[VetVerdict],
    candidate_id: str = "unknown",
) -> str:
    """Render vetting verdicts as a Markdown checklist.

    Args:
        verdicts: List from :func:`vet_candidate`.
        candidate_id: Used in the report header.

    Returns:
        Markdown string.
    """
    n_fail  = sum(1 for v in verdicts if v.verdict == "fail")
    n_warn  = sum(1 for v in verdicts if v.verdict == "warn")
    n_pass  = sum(1 for v in verdicts if v.verdict == "pass")
    n_miss  = sum(1 for v in verdicts if v.verdict == "missing")

    lines = [
        f"# Vetting Report — {candidate_id}",
        "",
        f"**FAIL:** {n_fail}  |  **WARN:** {n_warn}  "
        f"|  **PASS:** {n_pass}  |  **MISSING:** {n_miss}",
        "",
        "## FP Indicators",
        "| Feature | Score | Verdict |",
        "| --- | --- | --- |",
    ]
    for v in verdicts:
        if not v.is_fp_indicator:
            continue
        score_str = f"{v.score:.3f}" if v.score is not None else "—"
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "missing": "·"}[v.verdict]
        lines.append(f"| {v.feature_name} | {score_str} | {icon} {v.verdict} |")

    lines += [
        "",
        "## Quality Scores",
        "| Feature | Score | Verdict |",
        "| --- | --- | --- |",
    ]
    for v in verdicts:
        if v.is_fp_indicator:
            continue
        score_str = f"{v.score:.3f}" if v.score is not None else "—"
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "missing": "·"}[v.verdict]
        lines.append(f"| {v.feature_name} | {score_str} | {icon} {v.verdict} |")

    missing = [v for v in verdicts if v.verdict == "missing"]
    if missing:
        lines += [
            "",
            "## Missing Diagnostics",
            "| Feature | Reason |",
            "| --- | --- |",
        ]
        for v in missing:
            reason = v.missing_reason or "required diagnostic was not present"
            lines.append(f"| {v.feature_name} | {reason} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="false_positive_vetter",
        description="Structured FP vetting checklist for pipeline candidates.",
    )
    parser.add_argument("input", type=Path, metavar="FILE",
                        help="Pipeline JSON output file.")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Write Markdown report to this file.")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text())
    rows = data if isinstance(data, list) else [data]

    reports = []
    for row in rows:
        cid = row.get("candidate_id", "unknown")
        verdicts = vet_candidate(row)
        reports.append(format_vetting_report(verdicts, cid))

    combined = "\n---\n\n".join(reports)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(combined)
        print(f"Report written to {args.output}")
    else:
        print(combined, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
