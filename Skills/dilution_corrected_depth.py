"""Correct a measured transit depth for dilution from contaminating flux."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DilutionCorrectionResult:
    measured_depth_ppm: float
    dilution_fraction: float        # fraction of flux from contaminating source(s)
    corrected_depth_ppm: float      # depth_true = depth_obs / (1 - dilution)
    depth_ratio: float              # corrected / measured
    depth_err_ppm: float | None     # propagated error if input error given
    flag: str


def correct_for_dilution(
    measured_depth_ppm: float,
    dilution_fraction: float,
    depth_err_ppm: float | None = None,
) -> DilutionCorrectionResult:
    """
    Correct measured transit depth for flux dilution.

    If a fraction f of the total flux comes from a contaminating source,
    the true depth is: d_true = d_obs / (1 - f)

    Parameters
    ----------
    measured_depth_ppm: Observed transit depth in ppm.
    dilution_fraction:  Fraction of in-aperture flux from contaminating stars
                        (0 = no contamination, 1 = all flux from contaminant).
    depth_err_ppm:      Optional measurement uncertainty on depth (ppm).
    """
    if not math.isfinite(measured_depth_ppm) or measured_depth_ppm < 0:
        return DilutionCorrectionResult(
            measured_depth_ppm=measured_depth_ppm, dilution_fraction=dilution_fraction,
            corrected_depth_ppm=float("nan"), depth_ratio=float("nan"),
            depth_err_ppm=None, flag="INVALID_DEPTH",
        )
    if not math.isfinite(dilution_fraction) or dilution_fraction < 0 or dilution_fraction >= 1.0:
        return DilutionCorrectionResult(
            measured_depth_ppm=measured_depth_ppm, dilution_fraction=dilution_fraction,
            corrected_depth_ppm=float("nan"), depth_ratio=float("nan"),
            depth_err_ppm=None, flag="INVALID_DILUTION",
        )

    factor = 1.0 / (1.0 - dilution_fraction)
    corrected = measured_depth_ppm * factor
    ratio = factor

    err_out: float | None = None
    if depth_err_ppm is not None and math.isfinite(depth_err_ppm) and depth_err_ppm >= 0:
        err_out = round(depth_err_ppm * factor, 4)

    return DilutionCorrectionResult(
        measured_depth_ppm=measured_depth_ppm,
        dilution_fraction=dilution_fraction,
        corrected_depth_ppm=round(corrected, 4),
        depth_ratio=round(ratio, 6),
        depth_err_ppm=err_out,
        flag="OK",
    )


def format_dilution_result(r: DilutionCorrectionResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.4f}" if math.isfinite(v) else "N/A"

    err_str = f"{r.depth_err_ppm:.4f}" if r.depth_err_ppm is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Measured depth (ppm) | {_f(r.measured_depth_ppm)} |\n"
        f"| Dilution fraction | {_f(r.dilution_fraction)} |\n"
        f"| Corrected depth (ppm) | {_f(r.corrected_depth_ppm)} |\n"
        f"| Depth ratio (corr/meas) | {_f(r.depth_ratio)} |\n"
        f"| Depth error (ppm) | {err_str} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Correct transit depth for dilution.")
    p.add_argument("measured_depth_ppm", type=float)
    p.add_argument("dilution_fraction", type=float)
    p.add_argument("--depth-err-ppm", type=float, default=None)
    args = p.parse_args()
    r = correct_for_dilution(args.measured_depth_ppm, args.dilution_fraction, args.depth_err_ppm)
    print(format_dilution_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
