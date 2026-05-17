"""Predict future transit windows for a candidate signal.

Given an orbital period and reference epoch, computes the next N transit
mid-times with propagated timing uncertainty.

Public API
----------
predict_transits(period_days, epoch_bjd, n, *, period_err, epoch_err, t_start)
    -> list[TransitWindow]
format_transit_table(windows) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TransitWindow:
    transit_number: int
    mid_bjd: float
    window_start: float
    window_end: float
    uncertainty_hours: float


def predict_transits(
    period_days: float,
    epoch_bjd: float,
    n: int = 10,
    *,
    period_err: float | None = None,
    epoch_err: float | None = None,
    t_start: float | None = None,
    window_pad_hours: float = 1.0,
) -> list[TransitWindow]:
    """Predict N transit windows starting at or after t_start.

    Args:
        period_days: Orbital period in days.
        epoch_bjd: Reference transit mid-time (BJD).
        n: Number of future windows to predict.
        period_err: 1-sigma period uncertainty in days.
        epoch_err: 1-sigma epoch uncertainty in days.
        t_start: Start BJD for predictions (default: epoch_bjd).
        window_pad_hours: Fixed padding added to each side of the window (hours).

    Returns:
        List of :class:`TransitWindow` objects sorted by mid_bjd.

    Raises:
        ValueError: If period_days ≤ 0 or n ≤ 0.
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    t_ref = t_start if t_start is not None else epoch_bjd
    n_start = math.ceil((t_ref - epoch_bjd) / period_days)

    _pe = period_err if period_err is not None else 0.0
    _ee = epoch_err if epoch_err is not None else 0.0

    windows: list[TransitWindow] = []
    for i in range(n):
        k = n_start + i
        mid = epoch_bjd + k * period_days
        unc_days = math.sqrt(_ee ** 2 + (abs(k) * _pe) ** 2)
        unc_hours = unc_days * 24.0
        half = unc_days + window_pad_hours / 24.0
        windows.append(TransitWindow(
            transit_number=k,
            mid_bjd=mid,
            window_start=mid - half,
            window_end=mid + half,
            uncertainty_hours=unc_hours,
        ))

    return windows


def format_transit_table(windows: list[TransitWindow]) -> str:
    """Format transit windows as a Markdown table.

    Args:
        windows: List returned by :func:`predict_transits`.

    Returns:
        Markdown string, or ``"_No transits predicted._\\n"`` for an empty list.
    """
    if not windows:
        return "_No transits predicted._\n"

    header = "| # | Mid BJD | Window Start | Window End | Uncertainty (h) |"
    sep    = "| --- | --- | --- | --- | --- |"
    lines  = [header, sep]
    for w in windows:
        lines.append(
            f"| {w.transit_number} | {w.mid_bjd:.4f} "
            f"| {w.window_start:.4f} | {w.window_end:.4f} "
            f"| {w.uncertainty_hours:.2f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ephemeris_predictor",
        description="Predict future transit windows for a candidate signal.",
    )
    parser.add_argument("--period", type=float, required=True, metavar="DAYS",
                        help="Orbital period in days.")
    parser.add_argument("--epoch", type=float, required=True, metavar="BJD",
                        help="Reference transit epoch (BJD).")
    parser.add_argument("--n", type=int, default=10, metavar="N",
                        help="Number of windows to predict (default: 10).")
    parser.add_argument("--period-err", type=float, default=None, metavar="DAYS",
                        help="1-sigma period uncertainty in days.")
    parser.add_argument("--epoch-err", type=float, default=None, metavar="DAYS",
                        help="1-sigma epoch uncertainty in days.")
    parser.add_argument("--t-start", type=float, default=None, metavar="BJD",
                        help="Start BJD for predictions (default: epoch).")
    parser.add_argument("--pad-hours", type=float, default=1.0, metavar="H",
                        help="Fixed window padding in hours (default: 1.0).")
    args = parser.parse_args(argv)

    windows = predict_transits(
        args.period,
        args.epoch,
        args.n,
        period_err=args.period_err,
        epoch_err=args.epoch_err,
        t_start=args.t_start,
        window_pad_hours=args.pad_hours,
    )
    print(format_transit_table(windows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
