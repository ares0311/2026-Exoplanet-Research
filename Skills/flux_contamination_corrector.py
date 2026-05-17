"""Correct flux arrays for contamination using the CROWDSAP dilution factor.

The TESS CROWDSAP keyword gives the fraction of aperture flux that comes from
the target star.  Applying the correction recovers the target-only flux.

Public API
----------
ContaminationCorrectionResult(crowdsap, dilution_factor, flux_corrected,
                               depth_correction_factor, applied)
correct_flux_contamination(flux, *, crowdsap, contamination_ratio) -> ContaminationCorrectionResult
format_contamination_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContaminationCorrectionResult:
    crowdsap: float | None            # CROWDSAP value used (target fraction 0–1)
    dilution_factor: float            # = crowdsap (fraction from target)
    flux_corrected: tuple[float, ...] # corrected flux array
    depth_correction_factor: float    # multiply transit depths by this factor
    applied: bool


def correct_flux_contamination(
    flux: list[float],
    *,
    crowdsap: float | None = None,
    contamination_ratio: float | None = None,
) -> ContaminationCorrectionResult:
    """Remove neighbour contamination from a flux array.

    The corrected flux is:
        F_target(t) = (F_obs(t) - F_contam) / crowdsap
                    = F_obs(t) / crowdsap - (1 - crowdsap) / crowdsap

    where ``F_contam = (1 - crowdsap) * median(F_obs)``.

    Provide either ``crowdsap`` (TESS CROWDSAP header value, 0–1)
    or ``contamination_ratio`` (F_neighbours / F_target).

    Args:
        flux: Normalised flux array (median ≈ 1.0).
        crowdsap: Fraction of aperture flux from target star.
        contamination_ratio: Ratio of contaminating flux to target flux.

    Returns:
        :class:`ContaminationCorrectionResult`.
    """
    # Resolve crowdsap
    cs: float | None = None
    if crowdsap is not None:
        cs = max(0.0, min(1.0, float(crowdsap)))
    elif contamination_ratio is not None:
        cont = max(0.0, float(contamination_ratio))
        cs = 1.0 / (1.0 + cont) if cont >= 0 else None

    if cs is None or cs <= 0.0 or cs >= 1.0 - 1e-6:
        return ContaminationCorrectionResult(
            crowdsap=cs,
            dilution_factor=cs if cs is not None else 1.0,
            flux_corrected=tuple(float(f) for f in flux),
            depth_correction_factor=1.0,
            applied=False,
        )

    # Corrected flux: shift constant contamination floor
    # F_obs = cs * F_target + (1-cs) * 1.0   (normalized, contam level = 1)
    # F_target = (F_obs - (1-cs)) / cs
    contam_floor = 1.0 - cs
    corrected = [(float(f) - contam_floor) / cs for f in flux]
    depth_correction = 1.0 / cs

    return ContaminationCorrectionResult(
        crowdsap=cs,
        dilution_factor=cs,
        flux_corrected=tuple(corrected),
        depth_correction_factor=round(depth_correction, 6),
        applied=True,
    )


def format_contamination_result(result: ContaminationCorrectionResult) -> str:
    """Format contamination correction result as Markdown."""
    lines = [
        "## Flux Contamination Correction",
        "",
        f"- CROWDSAP: {result.crowdsap if result.crowdsap is not None else '—'}",
        f"- Dilution factor (target fraction): {result.dilution_factor:.4f}",
        f"- Depth correction factor: {result.depth_correction_factor:.4f}",
        f"- Correction applied: {result.applied}",
        f"- Corrected flux points: {len(result.flux_corrected)}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="flux_contamination_corrector",
        description="Correct flux arrays for CROWDSAP contamination.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--crowdsap", type=float, default=None)
    group.add_argument("--contamination-ratio", type=float, default=None)
    parser.add_argument("--output", default=None, metavar="JSON")
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = correct_flux_contamination(
        lc["flux"],
        crowdsap=args.crowdsap,
        contamination_ratio=args.contamination_ratio,
    )
    print(format_contamination_result(result))
    if args.output:
        out = dict(lc)
        out["flux"] = list(result.flux_corrected)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Corrected LC written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
