"""Compute predicted transit search window accounting for ephemeris uncertainty."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitSearchWindow:
    epoch_number: int
    predicted_time_bjd: float
    window_half_width_hours: float    # ±N_sigma window half-width
    window_start_bjd: float
    window_end_bjd: float
    sigma_total_hours: float          # total timing uncertainty at this epoch


@dataclass(frozen=True)
class TransitSearchWindowResult:
    windows: tuple[TransitSearchWindow, ...]
    n_sigma: float
    flag: str


def compute_search_window(
    epoch_bjd: float,
    period_days: float,
    sigma_period_days: float,
    n_future_epochs: int = 5,
    sigma_epoch_days: float = 0.0,
    n_sigma: float = 3.0,
) -> TransitSearchWindowResult:
    """Compute transit search windows with linear ephemeris uncertainty propagation.

    σ_window(n) = sqrt(σ_T0² + (n × σ_P)²)
    Window: [t_pred - n_sigma × σ_window, t_pred + n_sigma × σ_window]

    Args:
        epoch_bjd: reference transit epoch (BJD)
        period_days: orbital period (days)
        sigma_period_days: uncertainty on period (days)
        n_future_epochs: number of future transit windows to compute
        sigma_epoch_days: uncertainty on reference epoch (days)
        n_sigma: window half-width in sigma units
    """
    if period_days <= 0.0:
        return TransitSearchWindowResult((), n_sigma, "INVALID_PERIOD")
    if sigma_period_days < 0.0:
        return TransitSearchWindowResult((), n_sigma, "INVALID_SIGMA_PERIOD")
    if n_future_epochs < 1:
        return TransitSearchWindowResult((), n_sigma, "INVALID_N_EPOCHS")

    windows: list[TransitSearchWindow] = []
    for i in range(1, n_future_epochs + 1):
        t_pred = epoch_bjd + i * period_days
        sigma_total_days = math.sqrt(sigma_epoch_days**2 + (i * sigma_period_days)**2)
        sigma_total_hours = sigma_total_days * 24.0
        hw_hours = n_sigma * sigma_total_hours

        windows.append(TransitSearchWindow(
            epoch_number=i,
            predicted_time_bjd=t_pred,
            window_half_width_hours=hw_hours,
            window_start_bjd=t_pred - n_sigma * sigma_total_days,
            window_end_bjd=t_pred + n_sigma * sigma_total_days,
            sigma_total_hours=sigma_total_hours,
        ))

    return TransitSearchWindowResult(
        windows=tuple(windows),
        n_sigma=n_sigma,
        flag="OK",
    )


def format_search_window_result(r: TransitSearchWindowResult) -> str:
    if r.flag != "OK":
        return f"TransitSearchWindow | flag={r.flag}"
    lines = [
        f"Transit search windows (±{r.n_sigma:.1f}σ) | flag={r.flag}",
        "",
        "| Epoch | Predicted BJD | σ_window (hr) | Window (hr) |",
        "|---|---|---|---|",
    ]
    for w in r.windows:
        lines.append(
            f"| {w.epoch_number} "
            f"| {w.predicted_time_bjd:.4f} "
            f"| {w.sigma_total_hours:.2f} "
            f"| ±{w.window_half_width_hours:.2f} |"
        )
    return "\n".join(lines)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit search window calculator")
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("period_days", type=float)
    p.add_argument("sigma_period_days", type=float)
    p.add_argument("--n-epochs", type=int, default=5)
    args = p.parse_args()
    r = compute_search_window(args.epoch_bjd, args.period_days, args.sigma_period_days,
                               n_future_epochs=args.n_epochs)
    print(format_search_window_result(r))


if __name__ == "__main__":
    _cli()
