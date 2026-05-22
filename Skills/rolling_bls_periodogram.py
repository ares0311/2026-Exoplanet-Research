"""Rolling-window BLS periodogram to test signal persistence across time.

Splits the light curve into overlapping time windows and runs a lightweight
BLS period search in each window.  If the same period is recovered in most
windows the signal is persistent; a period recovered in only some windows
suggests a transient artifact.

Public API
----------
RollingBLSResult(period_days, n_windows, n_recovered, recovery_fraction,
                 recovered_periods, is_persistent, flag)
run_rolling_bls(time, flux, period_days, *, flux_err, window_days,
                step_days, period_tolerance_frac, n_durations,
                min_windows) -> RollingBLSResult
format_rolling_bls_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingBLSResult:
    period_days: float
    n_windows: int
    n_recovered: int
    recovery_fraction: float      # n_recovered / n_windows
    recovered_periods: tuple[float, ...]
    is_persistent: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _simple_bls_power(
    time: list[float],
    flux: list[float],
    period: float,
    duration_frac: float,
) -> float:
    """Compute BLS power for a single period/duration combination."""
    n = len(time)
    if n < 3:
        return 0.0
    mean_flux = sum(flux) / n

    best = 0.0
    n_phase = 20
    for i in range(n_phase):
        phase0 = i / n_phase
        in_flux: list[float] = []
        out_flux: list[float] = []
        for t, f in zip(time, flux, strict=False):
            ph = ((t / period) % 1.0)
            ph_diff = min(abs(ph - phase0), 1.0 - abs(ph - phase0))
            if ph_diff <= duration_frac / 2.0:
                in_flux.append(f)
            else:
                out_flux.append(f)
        if len(in_flux) < 2 or not out_flux:
            continue
        in_mean = sum(in_flux) / len(in_flux)
        out_mean = sum(out_flux) / len(out_flux)
        depth = out_mean - in_mean
        r = len(in_flux) / n
        power = depth ** 2 * r * (1 - r)
        if power > best:
            best = power
    return best - (mean_flux ** 2) * 0.0  # normalise reference


def _bls_best_period(
    time: list[float],
    flux: list[float],
    period_min: float,
    period_max: float,
    n_periods: int,
    n_durations: int,
) -> float:
    """Return the best-fit period from a coarse BLS grid."""
    if period_max <= period_min or n_periods < 2:
        return period_min
    step = (period_max - period_min) / (n_periods - 1)
    best_power = -1.0
    best_period = period_min
    duration_fracs = [0.02 * (k + 1) for k in range(n_durations)]
    for i in range(n_periods):
        p = period_min + i * step
        for df in duration_fracs:
            power = _simple_bls_power(time, flux, p, df)
            if power > best_power:
                best_power = power
                best_period = p
    return best_period


def run_rolling_bls(
    time: list[float],
    flux: list[float],
    period_days: float,
    *,
    flux_err: list[float] | None = None,
    window_days: float = 14.0,
    step_days: float = 7.0,
    period_tolerance_frac: float = 0.05,
    n_durations: int = 3,
    min_windows: int = 3,
    persistence_threshold: float = 0.6,
) -> RollingBLSResult:
    """Run rolling BLS to check whether a period is persistent.

    Args:
        time: Time array (days, BJD or similar).
        flux: Normalised flux array.
        period_days: Reference period to test for persistence.
        flux_err: Per-point uncertainties (unused currently, reserved).
        window_days: Width of each rolling time window in days.
        step_days: Step size between window starts.
        period_tolerance_frac: Fractional tolerance for period recovery
            (recovered if |p_found - period_days| / period_days <= tol).
        n_durations: Number of trial transit durations per period.
        min_windows: Minimum windows required to produce an OK result.
        persistence_threshold: Minimum recovery fraction for ``is_persistent``.

    Returns:
        :class:`RollingBLSResult`.
    """
    n = len(flux)
    if n < 5 or len(time) != n or period_days <= 0:
        return RollingBLSResult(period_days, 0, 0, 0.0, (), False, "INVALID")

    t_min = min(time)
    t_max = max(time)

    # Period search range: ±50% around reference
    p_lo = period_days * 0.5
    p_hi = period_days * 1.5
    n_periods = max(10, int((p_hi - p_lo) / (period_days * 0.01)) + 1)

    windows_starts: list[float] = []
    s = t_min
    while s + window_days <= t_max + 1e-9:
        windows_starts.append(s)
        s += step_days

    if len(windows_starts) < min_windows:
        return RollingBLSResult(period_days, 0, 0, 0.0, (), False, "INSUFFICIENT")

    recovered_periods: list[float] = []
    n_recovered = 0

    tol = period_days * period_tolerance_frac

    for ws in windows_starts:
        we = ws + window_days
        t_win: list[float] = []
        f_win: list[float] = []
        for t, f in zip(time, flux, strict=False):
            if ws <= t < we:
                t_win.append(t)
                f_win.append(f)
        if len(t_win) < 5:
            continue
        best_p = _bls_best_period(t_win, f_win, p_lo, p_hi, n_periods, n_durations)
        recovered_periods.append(round(best_p, 5))
        if abs(best_p - period_days) <= tol:
            n_recovered += 1

    n_windows = len(recovered_periods)
    if n_windows == 0:
        return RollingBLSResult(period_days, 0, 0, 0.0, (), False, "INSUFFICIENT")

    recovery_frac = n_recovered / n_windows
    is_persistent = recovery_frac >= persistence_threshold

    return RollingBLSResult(
        period_days=period_days,
        n_windows=n_windows,
        n_recovered=n_recovered,
        recovery_fraction=round(recovery_frac, 4),
        recovered_periods=tuple(recovered_periods),
        is_persistent=is_persistent,
        flag="OK",
    )


def format_rolling_bls_result(result: RollingBLSResult) -> str:
    """Format rolling BLS result as Markdown."""
    lines = [
        "## Rolling BLS Periodogram",
        "",
        f"- Reference period: {result.period_days:.4f} days",
        f"- Windows analysed: {result.n_windows}",
        f"- Windows recovered: {result.n_recovered}",
        f"- Recovery fraction: {result.recovery_fraction:.2%}",
        f"- Persistent: {'Yes' if result.is_persistent else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rolling_bls_periodogram",
        description="Test signal persistence via rolling BLS.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("--window-days", type=float, default=14.0)
    parser.add_argument("--step-days", type=float, default=7.0)
    args = parser.parse_args(argv)

    result = run_rolling_bls(
        [], [], args.period_days,
        window_days=args.window_days,
        step_days=args.step_days,
    )
    print(format_rolling_bls_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
