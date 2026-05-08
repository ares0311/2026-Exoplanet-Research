"""Injection-recovery completeness mapping for exoplanet transit searches.

Injects synthetic box transits into a real (or simulated) light curve, attempts
recovery via search_lightcurve, and maps the recovery rate as a function of
orbital period and transit depth.

Usage (standalone)
------------------
    python Skills/injection_recovery.py --tic 150428135 --periods 5 --depths 5

Usage (library)
---------------
    from Skills.injection_recovery import run_injection_recovery, InjectionGrid

    grid = run_injection_recovery(lc, target_id="TIC 150428135", mission="TESS")
    grid.print_summary()
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as a script from the repo root without pip install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.schemas import Mission
from exo_toolkit.search import search_lightcurve

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class InjectionResult:
    """Outcome of a single injection-recovery trial."""

    period_days: float
    depth_ppm: float
    duration_hours: float
    injected_epoch_bjd: float
    recovered: bool
    recovered_period_days: float | None
    recovered_depth_ppm: float | None
    recovered_snr: float | None


@dataclasses.dataclass
class InjectionGrid:
    """Aggregated recovery statistics over a period × depth grid."""

    period_grid: np.ndarray
    depth_grid: np.ndarray
    recovery_rate: np.ndarray  # shape (n_periods, n_depths), values in [0, 1]
    n_trials_per_cell: int
    results: list[InjectionResult]

    def print_summary(self) -> None:
        """Print a period × depth recovery-rate table to stdout."""
        depth_labels = [f"{d/1e3:.1f}k" for d in self.depth_grid]
        header = f"{'Period':>8s} | " + " | ".join(f"{d:>6s}" for d in depth_labels)
        print(header)
        print("-" * len(header))
        for i, period in enumerate(self.period_grid):
            row = f"{period:>7.1f}d | "
            row += " | ".join(
                f"{self.recovery_rate[i, j]:>5.0%} " for j in range(len(self.depth_grid))
            )
            print(row)


# ---------------------------------------------------------------------------
# Transit injection
# ---------------------------------------------------------------------------


def inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    depth_ppm: float,
) -> np.ndarray:
    """Return a copy of *flux* with a synthetic box transit injected.

    The transit is a simple flat-bottomed box: flux is multiplied by
    ``(1 - depth_ppm/1e6)`` during in-transit cadences.

    Args:
        time: Array of BJD timestamps.
        flux: Normalised flux array (median ≈ 1.0).
        period_days: Orbital period in days.
        epoch_bjd: Mid-transit time of first transit.
        duration_hours: Full transit duration in hours.
        depth_ppm: Transit depth in parts per million.

    Returns:
        New flux array with transit injected.
    """
    depth_frac = depth_ppm / 1e6
    half_dur = (duration_hours / 24.0) / 2.0
    phase = ((time - epoch_bjd) % period_days) / period_days
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    phase_days = phase * period_days
    in_transit = np.abs(phase_days) <= half_dur
    injected = flux.copy()
    injected[in_transit] *= 1.0 - depth_frac
    return injected


# ---------------------------------------------------------------------------
# Recovery check
# ---------------------------------------------------------------------------

_PERIOD_MATCH_RTOL = 0.02   # 2% relative tolerance for period match
_DEPTH_MATCH_RTOL = 0.50    # 50% relative tolerance for depth match (BLS is noisy)


def _is_recovered(
    injected_period: float,
    injected_depth: float,
    signals: list[Any],
) -> tuple[bool, Any | None]:
    """Check whether the injected signal appears in the recovered signal list."""
    for sig in signals:
        period_ok = abs(sig.period_days - injected_period) / injected_period <= _PERIOD_MATCH_RTOL
        # Also accept half/double period aliases
        half = injected_period / 2
        alias_half = abs(sig.period_days - half) / half <= _PERIOD_MATCH_RTOL
        double = injected_period * 2
        alias_double = abs(sig.period_days - double) / double <= _PERIOD_MATCH_RTOL
        depth_ok = abs(sig.depth_ppm - injected_depth) / injected_depth <= _DEPTH_MATCH_RTOL
        if (period_ok or alias_half or alias_double) and depth_ok:
            return True, sig
    return False, None


# ---------------------------------------------------------------------------
# Lightweight LC mock for standalone / test use
# ---------------------------------------------------------------------------


class _MockLC:
    """Minimal stand-in for a lightkurve LightCurve object."""

    def __init__(self, time_jd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> None:
        self.time = _MockTime(time_jd)
        self.flux = _MockArray(flux)
        self.flux_err = _MockArray(flux_err)


class _MockTime:
    def __init__(self, jd: np.ndarray) -> None:
        self.jd = jd


class _MockArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value


def make_mock_lc(
    baseline_days: float = 27.0,
    cadence_minutes: float = 2.0,
    noise_ppm: float = 500.0,
    rng: np.random.Generator | None = None,
) -> _MockLC:
    """Generate a synthetic TESS-like light curve for testing.

    Args:
        baseline_days: Total observation span in days.
        cadence_minutes: Cadence between exposures.
        noise_ppm: White-noise level in parts per million.
        rng: NumPy random generator for reproducibility.

    Returns:
        A mock LightCurve compatible with search_lightcurve.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = int(baseline_days * 24 * 60 / cadence_minutes)
    time = np.linspace(2458000.0, 2458000.0 + baseline_days, n)
    flux = 1.0 + rng.normal(0.0, noise_ppm / 1e6, n)
    flux_err = np.full(n, noise_ppm / 1e6)
    return _MockLC(time, flux, flux_err)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def run_injection_recovery(
    lc: Any,
    target_id: str,
    mission: Mission,
    *,
    period_grid: np.ndarray | None = None,
    depth_grid: np.ndarray | None = None,
    duration_hours: float = 2.0,
    n_trials: int = 3,
    min_snr: float = 5.0,
    rng: np.random.Generator | None = None,
) -> InjectionGrid:
    """Run a grid of injection-recovery trials and return completeness statistics.

    For each (period, depth) cell, *n_trials* trials are run with randomly
    drawn injection epochs. Recovery is declared when search_lightcurve returns
    a signal whose period and depth match within tolerances.

    Args:
        lc: A lightkurve-compatible LightCurve object.
        target_id: Target identifier string.
        mission: "TESS", "Kepler", or "K2".
        period_grid: 1-D array of trial periods in days.
        depth_grid: 1-D array of trial depths in ppm.
        duration_hours: Fixed transit duration injected for all trials (hours).
        n_trials: Trials per (period, depth) cell.
        min_snr: SNR threshold passed to search_lightcurve.
        rng: Random generator for reproducible epoch draws.

    Returns:
        InjectionGrid with per-cell recovery rates and raw trial results.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    time = lc.time.jd
    flux = lc.flux.value

    baseline = float(time[-1] - time[0])

    if period_grid is None:
        period_grid = np.array([1.0, 3.0, 7.0, 14.0, 28.0])
    if depth_grid is None:
        depth_grid = np.array([500.0, 1000.0, 2500.0, 5000.0, 10000.0])

    n_p, n_d = len(period_grid), len(depth_grid)
    recovery_count = np.zeros((n_p, n_d), dtype=int)
    all_results: list[InjectionResult] = []

    for i, period in enumerate(period_grid):
        if period >= baseline / 2:
            # Need at least 2 transits; skip this period for this baseline
            all_results.extend(
                InjectionResult(
                    period_days=period,
                    depth_ppm=float(depth),
                    duration_hours=duration_hours,
                    injected_epoch_bjd=float(time[0]),
                    recovered=False,
                    recovered_period_days=None,
                    recovered_depth_ppm=None,
                    recovered_snr=None,
                )
                for depth in depth_grid
            )
            continue

        for j, depth in enumerate(depth_grid):
            for _ in range(n_trials):
                epoch = float(rng.uniform(time[0], time[0] + period))
                injected_flux = inject_box_transit(
                    time, flux, period, epoch, duration_hours, float(depth)
                )

                mock = _MockLC(time, injected_flux, lc.flux_err.value)
                try:
                    signals = search_lightcurve(
                        mock,
                        target_id=target_id,
                        mission=mission,
                        period_min=max(0.5, period * 0.5),
                        period_max=min(baseline / 2.0, period * 2.0),
                        min_snr=min_snr,
                        max_peaks=3,
                    )
                except Exception:
                    signals = []

                recovered, best = _is_recovered(float(period), float(depth), signals)
                if recovered:
                    recovery_count[i, j] += 1

                all_results.append(
                    InjectionResult(
                        period_days=float(period),
                        depth_ppm=float(depth),
                        duration_hours=duration_hours,
                        injected_epoch_bjd=epoch,
                        recovered=recovered,
                        recovered_period_days=best.period_days if best else None,
                        recovered_depth_ppm=best.depth_ppm if best else None,
                        recovered_snr=best.snr if best else None,
                    )
                )

    recovery_rate = recovery_count / max(n_trials, 1)
    return InjectionGrid(
        period_grid=period_grid,
        depth_grid=depth_grid,
        recovery_rate=recovery_rate,
        n_trials_per_cell=n_trials,
        results=all_results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Injection-recovery completeness mapping for exoplanet transit searches."
    )
    p.add_argument("--tic", default="TIC 999999999", help="Target ID (default: synthetic LC)")
    p.add_argument("--mission", default="TESS", choices=["TESS", "Kepler", "K2"])
    p.add_argument("--periods", type=int, default=5, help="Number of period grid points")
    p.add_argument("--depths", type=int, default=5, help="Number of depth grid points")
    p.add_argument("--trials", type=int, default=3, help="Trials per grid cell")
    p.add_argument("--baseline", type=float, default=27.0, help="Synthetic LC baseline in days")
    p.add_argument("--noise", type=float, default=500.0, help="Synthetic LC noise in ppm")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    rng = np.random.default_rng(args.seed)

    print(f"Building synthetic light curve: baseline={args.baseline}d, noise={args.noise}ppm")
    lc = make_mock_lc(baseline_days=args.baseline, noise_ppm=args.noise, rng=rng)

    period_grid = np.geomspace(1.0, args.baseline / 3.0, args.periods)
    depth_grid = np.geomspace(500.0, 10000.0, args.depths)

    print(
        f"Running injection-recovery "
        f"({args.periods}×{args.depths} grid, {args.trials} trials/cell)…"
    )
    grid = run_injection_recovery(
        lc,
        target_id=args.tic,
        mission=args.mission,  # type: ignore[arg-type]
        period_grid=period_grid,
        depth_grid=depth_grid,
        n_trials=args.trials,
        rng=rng,
    )

    print("\n--- Recovery Rate (period rows × depth columns) ---")
    grid.print_summary()
    total = sum(r.recovered for r in grid.results)
    n = len(grid.results)
    print(f"\nOverall: {total}/{n} recovered ({total/max(n, 1):.0%})")


if __name__ == "__main__":
    main()
