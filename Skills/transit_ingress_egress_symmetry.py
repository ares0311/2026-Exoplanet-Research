"""Test ingress/egress symmetry from a phase-folded transit flux array."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IngressEgressSymmetryResult:
    n_in_transit: int
    ingress_depth_mean: float
    egress_depth_mean: float
    asymmetry: float
    asymmetry_sigma: float
    significance: float
    flag: str


def check_ingress_egress_symmetry(
    phase: list[float],
    flux: list[float],
    transit_half_width: float = 0.05,
    ingress_width: float = 0.01,
) -> IngressEgressSymmetryResult:
    """
    Compare ingress (phase < -mid) vs egress (phase > +mid) flux depth.

    Parameters
    ----------
    phase:             Phase values in [-0.5, 0.5).
    flux:              Normalised flux (1.0 = out-of-transit).
    transit_half_width: Half-width of full transit in phase units.
    ingress_width:      Width of ingress/egress zone in phase units.

    Asymmetry = mean(ingress_deficit) - mean(egress_deficit), where
    deficit = 1 - flux (positive when in transit).
    Significance = |asymmetry| / sigma_combined.
    """
    if len(phase) != len(flux):
        return IngressEgressSymmetryResult(
            n_in_transit=0, ingress_depth_mean=float("nan"),
            egress_depth_mean=float("nan"), asymmetry=float("nan"),
            asymmetry_sigma=float("nan"), significance=float("nan"),
            flag="LENGTH_MISMATCH",
        )
    if len(phase) < 4:
        return IngressEgressSymmetryResult(
            n_in_transit=0, ingress_depth_mean=float("nan"),
            egress_depth_mean=float("nan"), asymmetry=float("nan"),
            asymmetry_sigma=float("nan"), significance=float("nan"),
            flag="INSUFFICIENT_DATA",
        )

    hw = transit_half_width
    iw = ingress_width

    ingress_deficits = [
        1.0 - f for p, f in zip(phase, flux, strict=False)
        if -hw <= p <= -hw + iw
    ]
    egress_deficits = [
        1.0 - f for p, f in zip(phase, flux, strict=False)
        if hw - iw <= p <= hw
    ]

    n_in = len(ingress_deficits) + len(egress_deficits)
    if len(ingress_deficits) < 2 or len(egress_deficits) < 2:
        return IngressEgressSymmetryResult(
            n_in_transit=n_in, ingress_depth_mean=float("nan"),
            egress_depth_mean=float("nan"), asymmetry=float("nan"),
            asymmetry_sigma=float("nan"), significance=float("nan"),
            flag="INSUFFICIENT_INGRESS_EGRESS_POINTS",
        )

    ing_mean = sum(ingress_deficits) / len(ingress_deficits)
    egr_mean = sum(egress_deficits) / len(egress_deficits)
    asymmetry = ing_mean - egr_mean

    def _std(vals: list[float], mean: float) -> float:
        if len(vals) < 2:
            return float("nan")
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))

    sigma_ing = _std(ingress_deficits, ing_mean) / math.sqrt(len(ingress_deficits))
    sigma_egr = _std(egress_deficits, egr_mean) / math.sqrt(len(egress_deficits))
    sigma_combined = math.sqrt(sigma_ing**2 + sigma_egr**2)

    significance = abs(asymmetry) / sigma_combined if sigma_combined > 0 else 0.0

    flag = "ASYMMETRIC" if significance > 3.0 else "OK"

    return IngressEgressSymmetryResult(
        n_in_transit=n_in,
        ingress_depth_mean=round(ing_mean, 6),
        egress_depth_mean=round(egr_mean, 6),
        asymmetry=round(asymmetry, 6),
        asymmetry_sigma=round(sigma_combined, 6),
        significance=round(significance, 3),
        flag=flag,
    )


def format_symmetry_result(r: IngressEgressSymmetryResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N points (ingress+egress) | {r.n_in_transit} |\n"
        f"| Ingress mean depth | {r.ingress_depth_mean:.6f} |\n"
        f"| Egress mean depth | {r.egress_depth_mean:.6f} |\n"
        f"| Asymmetry | {r.asymmetry:.6f} |\n"
        f"| Significance (σ) | {r.significance:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check ingress/egress transit symmetry.")
    p.add_argument("phase_json", help="JSON array of phase values")
    p.add_argument("flux_json", help="JSON array of flux values")
    p.add_argument("--transit-half-width", type=float, default=0.05)
    p.add_argument("--ingress-width", type=float, default=0.01)
    args = p.parse_args()
    phase = json.loads(args.phase_json) if hasattr(args, "phase_json") else []
    flux = json.loads(args.flux_json) if hasattr(args, "flux_json") else []
    r = check_ingress_egress_symmetry(phase, flux, args.transit_half_width, args.ingress_width)
    print(format_symmetry_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
