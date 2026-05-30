from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitCountResult:
    n_transits: int
    next_transit_bjd: float
    window_days: float
    flag: str


def predict_transit_count(
    epoch_bjd: float,
    period_days: float,
    window_start_bjd: float,
    window_end_bjd: float,
) -> TransitCountResult:
    """Predict the number of transits within a BJD observation window.

    Finds the first transit at or after window_start_bjd and counts all
    transits up to and including window_end_bjd.
    """
    if period_days <= 0:
        return TransitCountResult(
            n_transits=0,
            next_transit_bjd=epoch_bjd,
            window_days=0.0,
            flag="INVALID_PERIOD",
        )

    if window_end_bjd <= window_start_bjd:
        return TransitCountResult(
            n_transits=0,
            next_transit_bjd=epoch_bjd,
            window_days=0.0,
            flag="INVALID_WINDOW",
        )

    window_days = window_end_bjd - window_start_bjd

    # Find the transit cycle number for the first transit in the window
    n_first = math.ceil((window_start_bjd - epoch_bjd) / period_days)
    next_transit_bjd = epoch_bjd + n_first * period_days

    # Count transits within the window
    n_transits = 0
    transit_time = next_transit_bjd
    while transit_time <= window_end_bjd:
        n_transits += 1
        transit_time += period_days

    # If no transits found, next_transit_bjd may be outside window — keep it
    # as the next upcoming transit after the window start
    return TransitCountResult(
        n_transits=n_transits,
        next_transit_bjd=next_transit_bjd,
        window_days=window_days,
        flag="OK",
    )


def format_transit_count(result: TransitCountResult) -> str:
    """Return a Markdown table summarising the transit count prediction."""
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| N Transits | {result.n_transits} |",
        f"| Next Transit (BJD) | {result.next_transit_bjd:.6f} |",
        f"| Window (days) | {result.window_days:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Predict the number of transits within an observation window."
    )
    parser.add_argument("epoch_bjd", type=float, help="Transit epoch in BJD.")
    parser.add_argument("period_days", type=float, help="Orbital period in days.")
    parser.add_argument(
        "window_start_bjd", type=float, help="Observation window start in BJD."
    )
    parser.add_argument(
        "window_end_bjd", type=float, help="Observation window end in BJD."
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    result = predict_transit_count(
        args.epoch_bjd,
        args.period_days,
        args.window_start_bjd,
        args.window_end_bjd,
    )

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_transit_count(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
