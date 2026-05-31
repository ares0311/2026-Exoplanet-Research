"""Plan optimal RV observation windows at quadrature phases."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RVWindow:
    phase: float
    bjd: float
    rv_fraction: float  # fraction of K amplitude (sin of phase angle)
    label: str


@dataclass(frozen=True)
class RVPlanResult:
    period_days: float
    epoch_bjd: float
    n_windows: int
    windows: tuple[RVWindow, ...]
    flag: str


def plan_rv_windows(
    period_days: float,
    epoch_bjd: float,
    start_bjd: float,
    n_windows: int = 4,
    *,
    include_transit: bool = False,
) -> RVPlanResult:
    """
    Return BJDs of optimal RV quadrature phases (φ = 0.25, 0.75) within
    a window starting at start_bjd, cycling over n_windows orbits.
    """
    if not math.isfinite(period_days) or period_days <= 0.0:
        return RVPlanResult(
            period_days=period_days, epoch_bjd=epoch_bjd,
            n_windows=0, windows=(), flag="INVALID_PERIOD",
        )
    if not math.isfinite(epoch_bjd):
        return RVPlanResult(
            period_days=period_days, epoch_bjd=epoch_bjd,
            n_windows=0, windows=(), flag="INVALID_EPOCH",
        )
    if n_windows < 1:
        return RVPlanResult(
            period_days=period_days, epoch_bjd=epoch_bjd,
            n_windows=0, windows=(), flag="INVALID_N_WINDOWS",
        )

    # Phases to sample: quadrature (0.25, 0.75) = max RV amplitude
    target_phases = [0.25, 0.75]
    if include_transit:
        target_phases = [0.0, 0.25, 0.5, 0.75]

    # Find the first orbit after start_bjd
    n_elapsed = math.ceil((start_bjd - epoch_bjd) / period_days)

    windows: list[RVWindow] = []
    orbit = n_elapsed
    collected = 0
    while collected < n_windows:
        for phase in target_phases:
            bjd = epoch_bjd + (orbit + phase) * period_days
            if bjd >= start_bjd:
                rv_frac = abs(math.sin(2.0 * math.pi * phase))
                if phase == 0.25:
                    label = "max_blueshift"
                elif phase == 0.75:
                    label = "max_redshift"
                elif phase == 0.0:
                    label = "transit"
                else:
                    label = "secondary"
                windows.append(RVWindow(
                    phase=phase,
                    bjd=round(bjd, 6),
                    rv_fraction=round(rv_frac, 4),
                    label=label,
                ))
                collected += 1
                if collected >= n_windows:
                    break
        orbit += 1

    return RVPlanResult(
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        n_windows=len(windows),
        windows=tuple(windows),
        flag="OK",
    )


def format_rv_plan(r: RVPlanResult) -> str:
    header = (
        "| BJD | Phase | RV fraction | Label |\n"
        "|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {w.bjd:.4f} | {w.phase:.2f} | {w.rv_fraction:.4f} | {w.label} |\n"
        for w in r.windows
    )
    return (
        f"**RV Window Plan** — P={r.period_days:.4f} d, T0={r.epoch_bjd:.4f}, "
        f"flag={r.flag}\n\n"
        f"{header}{rows}"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Plan RV quadrature observation windows.")
    p.add_argument("period_days", type=float)
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("start_bjd", type=float)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--include-transit", action="store_true")
    args = p.parse_args()
    r = plan_rv_windows(
        args.period_days, args.epoch_bjd, args.start_bjd,
        args.n_windows, include_transit=args.include_transit,
    )
    print(format_rv_plan(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
