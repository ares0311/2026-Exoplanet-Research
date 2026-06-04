"""Flag likely stellar multiples from RUWE, contrast limits, and separation."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_RUWE_THRESHOLD = 1.4        # Gaia RUWE above this → non-single star likely
_CONTRAST_THRESHOLD_MAG = 5.0  # contrast deeper than this rules out bright companion
_SEP_ARCSEC_TESS_PIXEL = 21.0  # TESS pixel scale in arcsec


@dataclass(frozen=True)
class MultiplicityResult:
    ruwe: float | None
    companion_sep_arcsec: float | None
    contrast_delta_mag: float | None
    ruwe_flag: bool
    separation_flag: bool
    contrast_flag: bool
    multiplicity_score: float
    verdict: str  # SINGLE | POSSIBLE_MULTIPLE | LIKELY_MULTIPLE
    flag: str


def check_stellar_multiplicity(
    ruwe: float | None = None,
    companion_sep_arcsec: float | None = None,
    contrast_delta_mag: float | None = None,
    depth_ppm: float | None = None,
) -> MultiplicityResult:
    """
    Flag likely stellar multiples from available diagnostics.

    ruwe: Gaia RUWE > 1.4 → astrometric excess noise → possible binary.
    companion_sep_arcsec: separation of nearest significant neighbour.
        If within 1 TESS pixel (21"), it dilutes the transit.
    contrast_delta_mag: magnitude difference to companion.
        If shallow (<5 mag), companion is bright enough to cause transit-like dip.
    depth_ppm: transit depth; very deep (>20000 ppm = 2%) may indicate EB.

    Multiplicity score ∈ [0, 1]: higher = more likely multiple.
    """
    evidence: list[float] = []
    ruwe_flag = False
    sep_flag = False
    contrast_flag = False

    if ruwe is not None and math.isfinite(ruwe):
        if ruwe > _RUWE_THRESHOLD:
            ruwe_flag = True
            score = min((ruwe - _RUWE_THRESHOLD) / 2.0, 1.0)
            evidence.append(score)
        else:
            evidence.append(0.0)

    if companion_sep_arcsec is not None and math.isfinite(companion_sep_arcsec):
        if companion_sep_arcsec < _SEP_ARCSEC_TESS_PIXEL:
            sep_flag = True
            score = 1.0 - companion_sep_arcsec / _SEP_ARCSEC_TESS_PIXEL
            evidence.append(score)
        else:
            evidence.append(0.0)

    if contrast_delta_mag is not None and math.isfinite(contrast_delta_mag):
        if contrast_delta_mag < _CONTRAST_THRESHOLD_MAG:
            contrast_flag = True
            score = 1.0 - contrast_delta_mag / _CONTRAST_THRESHOLD_MAG
            evidence.append(score)
        else:
            evidence.append(0.0)

    if depth_ppm is not None and math.isfinite(depth_ppm) and depth_ppm > 20000:
        evidence.append(min((depth_ppm - 20000) / 80000, 1.0))

    if not evidence:
        return MultiplicityResult(
            ruwe=ruwe, companion_sep_arcsec=companion_sep_arcsec,
            contrast_delta_mag=contrast_delta_mag,
            ruwe_flag=False, separation_flag=False, contrast_flag=False,
            multiplicity_score=0.0, verdict="SINGLE", flag="NO_DIAGNOSTICS",
        )

    mult_score = sum(evidence) / len(evidence)

    if mult_score >= 0.60:
        verdict = "LIKELY_MULTIPLE"
    elif mult_score >= 0.30:
        verdict = "POSSIBLE_MULTIPLE"
    else:
        verdict = "SINGLE"

    return MultiplicityResult(
        ruwe=ruwe,
        companion_sep_arcsec=companion_sep_arcsec,
        contrast_delta_mag=contrast_delta_mag,
        ruwe_flag=ruwe_flag,
        separation_flag=sep_flag,
        contrast_flag=contrast_flag,
        multiplicity_score=round(mult_score, 3),
        verdict=verdict,
        flag="OK",
    )


def format_multiplicity_result(r: MultiplicityResult) -> str:
    ruwe_str = f"{r.ruwe:.3f}" if r.ruwe is not None else "N/A"
    sep_str = f"{r.companion_sep_arcsec:.2f}" if r.companion_sep_arcsec is not None else "N/A"
    con_str = f"{r.contrast_delta_mag:.2f}" if r.contrast_delta_mag is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| RUWE | {ruwe_str} |\n"
        f"| Companion sep (arcsec) | {sep_str} |\n"
        f"| Contrast Δmag | {con_str} |\n"
        f"| RUWE flag | {r.ruwe_flag} |\n"
        f"| Separation flag | {r.separation_flag} |\n"
        f"| Contrast flag | {r.contrast_flag} |\n"
        f"| Multiplicity score | {r.multiplicity_score:.3f} |\n"
        f"| Verdict | {r.verdict} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check stellar multiplicity indicators.")
    p.add_argument("--ruwe", type=float, default=None)
    p.add_argument("--companion-sep-arcsec", type=float, default=None)
    p.add_argument("--contrast-delta-mag", type=float, default=None)
    p.add_argument("--depth-ppm", type=float, default=None)
    args = p.parse_args()
    r = check_stellar_multiplicity(
        args.ruwe, args.companion_sep_arcsec, args.contrast_delta_mag, args.depth_ppm
    )
    print(format_multiplicity_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
