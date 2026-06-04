"""Enumerate astrophysical false-positive scenarios with probability weights."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FpScenario:
    name: str
    description: str
    probability_weight: float
    key_diagnostic: str
    ruled_out: bool


@dataclass(frozen=True)
class FpScenarioResult:
    n_scenarios: int
    n_ruled_out: int
    total_fp_weight: float
    dominant_scenario: str
    scenarios: tuple[FpScenario, ...]
    flag: str


def build_fp_scenarios(
    candidate: dict,
) -> FpScenarioResult:
    """
    Enumerate astrophysical FP scenarios from available diagnostics.

    Scenarios:
    1. Background eclipsing binary (BEB): diluted EB behind target
    2. Hierarchical triple (HEB): bound companion with eclipsing secondary
    3. Grazing EB: target itself is an EB with grazing eclipse
    4. Stellar companion transit: transit of bound companion
    5. Instrumental artefact: systematic noise source

    Uses available diagnostics to rule out scenarios:
    - Centroid shift > 0.5": BEB likely
    - Centroid shift < 0.5" + deep transit: BEB less likely
    - Secondary eclipse: EB more likely
    - Odd/even asymmetry > 3σ: grazing EB or HEB
    - RUWE > 1.4: hierarchical triple candidate
    - Very deep transit (> 3%): grazing EB candidate
    - Low CROWDSAP: BEB more likely
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

    depth_ppm = _get("depth_ppm", _get("best_depth_ppm"))
    centroid = _get("centroid_motion_arcsec")
    secondary_snr = _get("secondary_snr", 0.0)
    odd_even_sigma = _get("odd_even_sigma", 0.0)
    ruwe = _get("ruwe")
    crowdsap = _get("crowdsap", _get("contamination_ratio"))
    snr = _get("snr", _get("best_snr", 5.0))

    # Initial probability weights (priors from occurrence statistics)
    # Rough: BEB ~20%, HEB ~15%, grazing EB ~10%, companion ~10%, artefact ~5% of all detections
    scenarios: list[dict] = [
        {
            "name": "BACKGROUND_EB",
            "description": "Background eclipsing binary diluted by target flux",
            "weight": 0.20,
            "diagnostic": "Centroid shift + CROWDSAP + galactic latitude",
            "ruled_out": False,
        },
        {
            "name": "HIERARCHICAL_TRIPLE",
            "description": "Bound companion pair causing secondary eclipse",
            "weight": 0.15,
            "diagnostic": "Gaia RUWE + AO imaging + secondary eclipse depth ratio",
            "ruled_out": False,
        },
        {
            "name": "GRAZING_EB",
            "description": "Eclipsing binary with grazing geometry (near-equal depth)",
            "weight": 0.10,
            "diagnostic": "Odd/even depth test + V-shaped transit + T14/T23 ratio",
            "ruled_out": False,
        },
        {
            "name": "STELLAR_COMPANION_TRANSIT",
            "description": "Transit of a physically bound stellar companion",
            "weight": 0.10,
            "diagnostic": "Deep transit depth (> 1%) + RV trend + adaptive optics",
            "ruled_out": False,
        },
        {
            "name": "INSTRUMENTAL_ARTEFACT",
            "description": "Systematic noise mimicking a transit signal",
            "weight": 0.05,
            "diagnostic": "Cross-sector consistency + centroid test + SNR vs single pixel",
            "ruled_out": False,
        },
    ]

    # Update weights and rule-out status based on diagnostics
    for sc in scenarios:
        if sc["name"] == "BACKGROUND_EB":
            if centroid is not None:
                if centroid > 1.0:
                    sc["weight"] = 0.70  # Strong BEB evidence
                elif centroid < 0.3:
                    sc["weight"] = 0.05
                    sc["ruled_out"] = True
            if crowdsap is not None and crowdsap < 0.5:
                sc["weight"] = min(1.0, sc["weight"] * 1.5)

        elif sc["name"] == "HIERARCHICAL_TRIPLE":
            if ruwe is not None and ruwe > 1.4:
                sc["weight"] = 0.35
            if secondary_snr is not None and secondary_snr > 3.0:
                sc["weight"] = min(1.0, sc["weight"] * 1.5)

        elif sc["name"] == "GRAZING_EB":
            if odd_even_sigma is not None and odd_even_sigma > 3.0:
                sc["weight"] = 0.50
            if depth_ppm is not None and depth_ppm > 30000:
                sc["weight"] = max(sc["weight"], 0.35)
            if secondary_snr is not None and secondary_snr < 1.0 and (
                odd_even_sigma is None or odd_even_sigma < 1.0
            ):
                sc["ruled_out"] = True
                sc["weight"] = 0.01

        elif sc["name"] == "STELLAR_COMPANION_TRANSIT":
            if depth_ppm is not None and depth_ppm > 10000:
                sc["weight"] = 0.25
            elif depth_ppm is not None and depth_ppm < 1000:
                sc["weight"] = 0.02

        elif sc["name"] == "INSTRUMENTAL_ARTEFACT":
            if snr is not None and snr > 15.0:
                sc["ruled_out"] = True
                sc["weight"] = 0.005

    # Normalise weights
    total = sum(sc["weight"] for sc in scenarios)
    if total > 0:
        for sc in scenarios:
            sc["weight"] = round(sc["weight"] / total, 4)

    sc_objects = tuple(
        FpScenario(
            name=sc["name"],
            description=sc["description"],
            probability_weight=sc["weight"],
            key_diagnostic=sc["diagnostic"],
            ruled_out=sc["ruled_out"],
        )
        for sc in sorted(scenarios, key=lambda s: s["weight"], reverse=True)
    )

    n_ruled = sum(1 for s in sc_objects if s.ruled_out)
    total_fp = sum(s.probability_weight for s in sc_objects)
    dominant = sc_objects[0].name if sc_objects else "UNKNOWN"

    return FpScenarioResult(
        n_scenarios=len(sc_objects),
        n_ruled_out=n_ruled,
        total_fp_weight=round(total_fp, 4),
        dominant_scenario=dominant,
        scenarios=sc_objects,
        flag="OK",
    )


def format_fp_scenarios(r: FpScenarioResult) -> str:
    lines = [
        f"**FP Scenario Builder** — {r.n_scenarios} scenarios, "
        f"{r.n_ruled_out} ruled out, dominant: {r.dominant_scenario}\n",
        "| Scenario | Weight | Ruled Out | Key Diagnostic |",
        "|---|---|---|---|",
    ]
    for s in r.scenarios:
        lines.append(
            f"| {s.name} | {s.probability_weight:.4f} | {s.ruled_out} | {s.key_diagnostic} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Build FP scenario list for a candidate.")
    p.add_argument("candidate_json", help="JSON dict or @file")
    args = p.parse_args()
    raw = args.candidate_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            cand = json.load(f)
    else:
        cand = json.loads(raw)
    r = build_fp_scenarios(cand)
    print(format_fp_scenarios(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
