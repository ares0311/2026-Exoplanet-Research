"""Generate a phase-folded light curve PNG from a candidate JSON row.

Reads a JSON file produced by ``exo <TIC-ID> --output results.json`` and
generates a phase-folded flux plot for each candidate signal.

Public API
----------
phase_fold(time, flux, period, epoch) -> tuple[np.ndarray, np.ndarray]
plot_candidate(row, *, output_dir, show) -> Path | None
plot_all(path, *, output_dir, show) -> list[Path]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Core phase-folding math (no matplotlib dependency)
# ---------------------------------------------------------------------------


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase, flux) where phase ∈ [-0.5, 0.5).

    Args:
        time: 1-D array of observation times (any consistent unit).
        flux: 1-D flux array aligned with ``time``.
        period: Orbital period in same units as ``time``.
        epoch: Transit midpoint reference time.

    Returns:
        Tuple of (phase, flux) sorted by phase, both 1-D arrays.
    """
    if period <= 0.0:
        raise ValueError(f"period must be > 0, got {period}")
    phase = ((time - epoch) % period) / period
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    order = np.argsort(phase)
    return phase[order], flux[order]


# ---------------------------------------------------------------------------
# Single-candidate plot
# ---------------------------------------------------------------------------


def plot_candidate(
    row: dict[str, Any],
    *,
    output_dir: Path | str = Path("."),
    show: bool = False,
    time: np.ndarray | None = None,
    flux: np.ndarray | None = None,
) -> Path | None:
    """Generate a phase-folded PNG for one candidate row.

    Args:
        row: Dict with at least ``candidate_id``, ``period_days``, ``epoch_bjd``.
        output_dir: Directory where the PNG is written.
        show: If True, call ``plt.show()`` after saving (blocks in interactive use).
        time: Optional preloaded time array (BJD). If None, plot is generated
              without data points (metadata-only preview).
        flux: Optional preloaded flux array. Required when ``time`` is provided.

    Returns:
        Path of the written PNG, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    candidate_id = str(row.get("candidate_id", "unknown"))
    period = float(row.get("period_days", 1.0))
    epoch = float(row.get("epoch_bjd", 0.0))
    depth_ppm = float(row.get("depth_ppm", 0.0))
    snr = float(row.get("snr", 0.0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{candidate_id}_phase_fold.png"

    fig, ax = plt.subplots(figsize=(8, 4))

    if time is not None and flux is not None:
        ph, fl = phase_fold(
            np.asarray(time, dtype=float), np.asarray(flux, dtype=float), period, epoch
        )
        ax.scatter(ph, fl, s=1, alpha=0.4, color="steelblue", label="data")

    ax.axvline(0.0, color="tomato", linewidth=1.2, linestyle="--", label="transit")
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Relative flux")
    title = (
        f"{candidate_id}  |  P = {period:.4f} d  |  "
        f"depth = {depth_ppm:.0f} ppm  |  SNR = {snr:.1f}"
    )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    if show:
        plt.show()

    return out_path


# ---------------------------------------------------------------------------
# Batch plot from JSON file
# ---------------------------------------------------------------------------


def plot_all(
    path: Path | str,
    *,
    output_dir: Path | str = Path("."),
    show: bool = False,
) -> list[Path]:
    """Plot every candidate in a JSON results file.

    Args:
        path: JSON file produced by ``exo --output``.
        output_dir: Directory for output PNGs.
        show: Passed to :func:`plot_candidate`.

    Returns:
        List of paths for written PNGs (empty if matplotlib unavailable).
    """
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else [data]
    out: list[Path] = []
    for row in rows:
        p = plot_candidate(row, output_dir=output_dir, show=show)
        if p is not None:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="plot_lc",
        description="Generate phase-folded light curve PNGs from exo JSON output.",
    )
    parser.add_argument(
        "file",
        type=Path,
        metavar="FILE",
        help="JSON file produced by `exo --output`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        metavar="DIR",
        help="Directory for output PNGs (default: plots/).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each plot interactively (blocks).",
    )
    args = parser.parse_args(argv)

    paths = plot_all(args.file, output_dir=args.output_dir, show=args.show)
    if not paths:
        print("No plots generated (matplotlib not installed?).")
        return 1
    for p in paths:
        print(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
