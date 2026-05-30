from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitShapeResult:
    shape: str
    u_score: float
    v_score: float
    flat_score: float
    flag: str


def classify_transit_shape(
    in_transit_flux: list[float],
    tolerance: float = 0.1,
) -> TransitShapeResult:
    """Classify the shape of a transit from normalised in-transit flux values.

    Flux values use OOT=0.0 convention, so in-transit values are negative.
    """
    if len(in_transit_flux) < 5:
        return TransitShapeResult(
            shape="unknown",
            u_score=0.0,
            v_score=0.0,
            flat_score=0.0,
            flag="INSUFFICIENT_DATA",
        )

    min_flux = min(in_transit_flux)
    depth = abs(min_flux)

    if depth == 0.0:
        return TransitShapeResult(
            shape="unknown",
            u_score=0.0,
            v_score=0.0,
            flat_score=0.0,
            flag="ZERO_DEPTH",
        )

    n = len(in_transit_flux)
    threshold = depth * (1.0 - tolerance)

    # v_score: fraction of points NOT at full depth (|flux| < threshold)
    v_count = sum(1 for f in in_transit_flux if abs(f) < threshold)
    v_score = v_count / n

    # flat_score: fraction of points near minimum (|flux| >= threshold)
    flat_count = sum(1 for f in in_transit_flux if abs(f) >= threshold)
    flat_score = flat_count / n

    # u_score: intermediate — neither fully V nor fully flat
    u_score = 1.0 - flat_score - v_score

    if flat_score > 0.6:
        shape = "flat-bottom"
    elif v_score > 0.6:
        shape = "V-shaped"
    elif u_score > 0.3:
        shape = "U-shaped"
    else:
        shape = "unknown"

    return TransitShapeResult(
        shape=shape,
        u_score=u_score,
        v_score=v_score,
        flat_score=flat_score,
        flag="OK",
    )


def format_transit_shape(result: TransitShapeResult) -> str:
    """Return a Markdown table summarising the transit shape classification."""
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Shape | {result.shape} |",
        f"| U-score | {result.u_score:.4f} |",
        f"| V-score | {result.v_score:.4f} |",
        f"| Flat-score | {result.flat_score:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Classify transit shape from in-transit flux values."
    )
    parser.add_argument(
        "flux",
        nargs="+",
        type=float,
        help="In-transit normalised flux values (OOT=0 convention, negatives).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Depth fraction tolerance for flat-bottom detection (default 0.1).",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    result = classify_transit_shape(args.flux, tolerance=args.tolerance)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_transit_shape(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
