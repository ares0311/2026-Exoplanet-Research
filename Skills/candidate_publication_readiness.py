"""Assess whether a candidate is ready for formal publication or TFOP submission."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

_GATE_ITEMS: list[tuple[str, str]] = [
    ("low_fpp", "FPP < 0.10"),
    ("high_dc", "Detection confidence > 0.80"),
    ("multi_transit", "≥ 3 independent transits"),
    ("acceptable_snr", "Transit SNR ≥ 7.5"),
    ("consistent_depth", "Depth consistent across sectors"),
    ("no_secondary", "No significant secondary eclipse"),
    ("no_odd_even", "Odd/even depths consistent"),
    ("centroid_ok", "Centroid shift < 1 arcsec"),
    ("stellar_params", "Stellar Teff/logg/radius available"),
    ("spectral_type_ok", "Host is FGK dwarf (logg > 4.0)"),
]


@dataclass(frozen=True)
class PublicationReadinessResult:
    n_gates: int
    n_passed: int
    readiness_score: float
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    recommendation: str   # READY / NEEDS_FOLLOW_UP / NOT_READY
    flag: str


def assess_publication_readiness(
    candidate: dict,
) -> PublicationReadinessResult:
    """
    Check a candidate dict against standard publication-readiness gates.

    Expects keys (all optional):
    - false_positive_probability (or scores.false_positive_probability)
    - detection_confidence (or scores.detection_confidence)
    - n_transits
    - snr / best_snr
    - depth_consistent (bool)
    - secondary_snr
    - odd_even_sigma
    - centroid_motion_arcsec
    - stellar_teff_k, stellar_logg, stellar_radius_rsun
    """
    scores = candidate.get("scores", {})

    def _get(key: str, default: float | None = None) -> float | None:
        v = candidate.get(key, scores.get(key, default))
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fpp = _get("false_positive_probability", _get("fpp"))
    dc = _get("detection_confidence")
    n_tr = _get("n_transits")
    snr = _get("snr", _get("best_snr"))
    depth_ok = candidate.get("depth_consistent")
    sec_snr = _get("secondary_snr", 0.0)
    oe_sigma = _get("odd_even_sigma", 0.0)
    centroid = _get("centroid_motion_arcsec")
    teff = _get("stellar_teff_k")
    logg = _get("stellar_logg")
    radius = _get("stellar_radius_rsun")

    passed: list[str] = []
    failed: list[str] = []

    checks = {
        "low_fpp": fpp is not None and fpp < 0.10,
        "high_dc": dc is not None and dc > 0.80,
        "multi_transit": n_tr is not None and n_tr >= 3,
        "acceptable_snr": snr is not None and snr >= 7.5,
        "consistent_depth": depth_ok is True or depth_ok == 1,
        "no_secondary": sec_snr is None or sec_snr < 3.0,
        "no_odd_even": oe_sigma is None or oe_sigma < 3.0,
        "centroid_ok": centroid is None or centroid < 1.0,
        "stellar_params": teff is not None and logg is not None and radius is not None,
        "spectral_type_ok": logg is not None and logg > 4.0 and (
            teff is None or 3500 < teff < 7500
        ),
    }

    for key, label in _GATE_ITEMS:
        result = checks.get(key, False)
        if result:
            passed.append(label)
        else:
            failed.append(label)

    n_passed = len(passed)
    n_gates = len(_GATE_ITEMS)
    score = n_passed / n_gates

    if score >= 0.90:
        rec = "READY"
    elif score >= 0.70:
        rec = "NEEDS_FOLLOW_UP"
    else:
        rec = "NOT_READY"

    return PublicationReadinessResult(
        n_gates=n_gates,
        n_passed=n_passed,
        readiness_score=round(score, 3),
        passed_gates=tuple(passed),
        failed_gates=tuple(failed),
        recommendation=rec,
        flag="OK",
    )


def format_readiness_result(r: PublicationReadinessResult) -> str:
    passed_str = "\n".join(f"  - ✓ {g}" for g in r.passed_gates) or "  (none)"
    failed_str = "\n".join(f"  - ✗ {g}" for g in r.failed_gates) or "  (none)"
    return (
        f"**Publication Readiness** — {r.n_passed}/{r.n_gates} gates passed "
        f"({r.readiness_score:.0%}) → **{r.recommendation}**\n\n"
        f"Passed:\n{passed_str}\n\n"
        f"Failed:\n{failed_str}\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Assess candidate publication readiness.")
    p.add_argument("candidate_json", help="JSON dict or @file")
    args = p.parse_args()
    raw = args.candidate_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            cand = json.load(f)
    else:
        cand = json.loads(raw)
    r = assess_publication_readiness(cand)
    print(format_readiness_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
