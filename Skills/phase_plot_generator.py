"""Generate phase-folded light curve plots for publication or reports.

Writes PNG files using matplotlib (optional import; returns None if absent).

Public API
----------
PhasePlotResult(tic_id, period_days, epoch_bjd, output_path, n_points, success)
generate_phase_plot(tic_id, time, flux, period, epoch, *, flux_err,
                    output_path, title, binned, bin_minutes,
                    plot_fn) -> PhasePlotResult
format_plot_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhasePlotResult:
    tic_id: int
    period_days: float
    epoch_bjd: float
    output_path: Path | None   # None if matplotlib absent or plot_fn returned None
    n_points: int
    success: bool
    message: str               # empty on success, error message on failure


def _phase_fold(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
) -> tuple[list[float], list[float]]:
    """Return (phase, flux) with phase in [-0.5, 0.5)."""
    phase = [((t - epoch) % period) / period for t in time]
    phase = [p - 1.0 if p >= 0.5 else p for p in phase]
    pairs = sorted(zip(phase, flux, strict=False), key=lambda x: x[0])
    ph, fl = zip(*pairs, strict=False) if pairs else ([], [])
    return list(ph), list(fl)


def _bin_phase(
    phase: list[float],
    flux: list[float],
    n_bins: int = 100,
) -> tuple[list[float], list[float]]:
    """Bin phase-folded flux into equal-width bins."""
    if not phase:
        return [], []
    bin_edges = [-0.5 + i / n_bins for i in range(n_bins + 1)]
    bin_centers: list[float] = []
    bin_flux: list[float] = []
    for j in range(n_bins):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        pts = [fl for ph, fl in zip(phase, flux, strict=False) if lo <= ph < hi]
        if pts:
            bin_centers.append((lo + hi) / 2.0)
            bin_flux.append(sum(pts) / len(pts))
    return bin_centers, bin_flux


def _default_plot_fn(
    tic_id: int,
    phase: list[float],
    flux: list[float],
    *,
    bin_phase: list[float],
    bin_flux: list[float],
    period_days: float,
    epoch_bjd: float,
    title: str,
    output_path: Path,
) -> Path | None:
    """Generate phase-folded plot using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(phase, flux, s=1, alpha=0.3, color="steelblue", label="data")
    if bin_phase:
        ax.plot(bin_phase, bin_flux, "r-", lw=1.5, label="binned")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux")
    ax.set_title(title or f"TIC {tic_id} — P={period_days:.4f} d")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def generate_phase_plot(
    tic_id: int,
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    output_path: Path | str | None = None,
    title: str = "",
    binned: bool = True,
    bin_minutes: float = 30.0,
    plot_fn=None,
) -> PhasePlotResult:
    """Generate a phase-folded light curve plot.

    Args:
        tic_id: TIC identifier (used in title and default filename).
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Reference mid-transit epoch (BJD).
        flux_err: Per-cadence uncertainties (unused in plot; reserved).
        output_path: Output PNG path.  Defaults to
            ``plots/TIC_{tic_id}_phase.png``.
        title: Plot title override.
        binned: If True, overlay a binned curve.
        bin_minutes: Bin width in minutes for the overlay.
        plot_fn: Injectable plotting function for testing.

    Returns:
        :class:`PhasePlotResult`.
    """
    if output_path is None:
        output_path = Path(f"plots/TIC_{tic_id}_phase.png")
    else:
        output_path = Path(output_path)

    n_points = len(time)

    if period <= 0 or n_points == 0:
        return PhasePlotResult(
            tic_id=tic_id, period_days=period, epoch_bjd=epoch,
            output_path=None, n_points=n_points, success=False,
            message="Invalid period or empty light curve.",
        )

    phase, f_folded = _phase_fold(time, flux, period, epoch)

    # Bin overlay
    bin_ph: list[float] = []
    bin_fl: list[float] = []
    if binned and phase:
        n_bins = max(10, int(period * 24 * 60 / max(bin_minutes, 1.0)))
        bin_ph, bin_fl = _bin_phase(phase, f_folded, n_bins=min(n_bins, 200))

    fn = plot_fn if plot_fn is not None else _default_plot_fn

    try:
        result_path = fn(
            tic_id, phase, f_folded,
            bin_phase=bin_ph, bin_flux=bin_fl,
            period_days=period, epoch_bjd=epoch,
            title=title, output_path=output_path,
        )
    except Exception as exc:
        return PhasePlotResult(
            tic_id=tic_id, period_days=period, epoch_bjd=epoch,
            output_path=None, n_points=n_points, success=False,
            message=str(exc),
        )

    if result_path is None:
        return PhasePlotResult(
            tic_id=tic_id, period_days=period, epoch_bjd=epoch,
            output_path=None, n_points=n_points, success=False,
            message="matplotlib not available",
        )

    return PhasePlotResult(
        tic_id=tic_id, period_days=period, epoch_bjd=epoch,
        output_path=result_path, n_points=n_points, success=True,
        message="",
    )


def format_plot_result(result: PhasePlotResult) -> str:
    """Format phase plot result as Markdown."""
    lines = [
        "## Phase Plot",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- Period: {result.period_days:.4f} d",
        f"- Epoch: {result.epoch_bjd:.4f} BJD",
        f"- Data points: {result.n_points}",
    ]
    if result.success and result.output_path is not None:
        lines.append(f"- Output: `{result.output_path}`")
    else:
        lines.append(f"- Status: {result.message or 'not generated'}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="phase_plot_generator",
        description="Generate phase-folded light curve plot.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    lc = json.loads(_Path(args.lc).read_text())
    result = generate_phase_plot(
        args.tic_id, lc["time"], lc["flux"],
        args.period, args.epoch,
        output_path=args.output,
    )
    print(format_plot_result(result))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
