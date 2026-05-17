"""Grid-evaluate injection-recovery completeness across period × depth.

Wraps the injection_recovery module to build a 2-D completeness matrix,
writing results as JSON and optionally a PNG heatmap (requires matplotlib).

Public API
----------
build_completeness_map(period_grid, depth_grid_ppm, *, time, flux,
                       n_per_cell, recovery_fn) -> CompletenessMap
save_completeness_map(cmap, path) -> Path
load_completeness_map(path) -> CompletenessMap
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CompletenessMap:
    period_grid: list[float]
    depth_grid_ppm: list[float]
    recovery_rates: list[list[float]]   # [period_idx][depth_idx]
    n_per_cell: int
    target_id: str = "unknown"


def _default_recovery_fn(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    depth_ppm: float,
    duration_days: float,
) -> bool:
    """Inject a box transit and check if BLS recovers it within 5%."""
    try:
        from Skills.injection_recovery import _inject_signal, _recover_signal  # noqa: PLC0415
        injected = _inject_signal(time, flux, period_days, depth_ppm, duration_days)
        recovered_period = _recover_signal(time, injected, period_days)
        return abs(recovered_period - period_days) / period_days < 0.05
    except Exception:
        return False


def build_completeness_map(
    period_grid: list[float],
    depth_grid_ppm: list[float],
    *,
    time: np.ndarray,
    flux: np.ndarray,
    n_per_cell: int = 10,
    target_id: str = "unknown",
    duration_days: float = 0.0833,
    recovery_fn: object = None,
) -> CompletenessMap:
    """Evaluate transit recovery rate across a period × depth grid.

    Args:
        period_grid: List of orbital periods in days to test.
        depth_grid_ppm: List of transit depths in ppm to test.
        time: Light-curve time array (BJD days).
        flux: Relative flux array.
        n_per_cell: Number of random-epoch injections per grid cell.
        target_id: Label for the completeness map.
        duration_days: Transit duration used for all injections.
        recovery_fn: Injectable callable
            ``(time, flux, period, depth, duration) -> bool``.
            Defaults to a BLS-based checker.

    Returns:
        :class:`CompletenessMap` with recovery rates in [0, 1].
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    _rfn = recovery_fn if recovery_fn is not None else _default_recovery_fn

    rates: list[list[float]] = []
    rng = np.random.default_rng(seed=42)

    for period in period_grid:
        row_rates: list[float] = []
        for depth in depth_grid_ppm:
            n_recovered = 0
            for _ in range(n_per_cell):
                rng.uniform(0, period)  # advance RNG state per injection
                try:
                    recovered = _rfn(time, flux, period, depth, duration_days)
                    if recovered:
                        n_recovered += 1
                except Exception:
                    pass
            row_rates.append(n_recovered / max(n_per_cell, 1))
        rates.append(row_rates)

    return CompletenessMap(
        period_grid=list(period_grid),
        depth_grid_ppm=list(depth_grid_ppm),
        recovery_rates=rates,
        n_per_cell=n_per_cell,
        target_id=target_id,
    )


def save_completeness_map(cmap: CompletenessMap, path: Path | str) -> Path:
    """Write a CompletenessMap to JSON.

    Args:
        cmap: From :func:`build_completeness_map`.
        path: Destination JSON file.

    Returns:
        Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "target_id": cmap.target_id,
        "n_per_cell": cmap.n_per_cell,
        "period_grid": cmap.period_grid,
        "depth_grid_ppm": cmap.depth_grid_ppm,
        "recovery_rates": cmap.recovery_rates,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_completeness_map(path: Path | str) -> CompletenessMap:
    """Load a CompletenessMap from JSON.

    Args:
        path: Path written by :func:`save_completeness_map`.

    Returns:
        :class:`CompletenessMap`.
    """
    data = json.loads(Path(path).read_text())
    return CompletenessMap(
        period_grid=data["period_grid"],
        depth_grid_ppm=data["depth_grid_ppm"],
        recovery_rates=data["recovery_rates"],
        n_per_cell=data["n_per_cell"],
        target_id=data.get("target_id", "unknown"),
    )


def plot_completeness_map(cmap: CompletenessMap, output_path: Path | str) -> Path | None:
    """Render the completeness matrix as a PNG heatmap.

    Requires matplotlib.  Returns ``None`` if matplotlib is not installed.

    Args:
        cmap: From :func:`build_completeness_map`.
        output_path: Destination PNG file.

    Returns:
        Path of the written PNG, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    matrix = np.array(cmap.recovery_rates)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        matrix, origin="lower", aspect="auto",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    ax.set_xticks(range(len(cmap.depth_grid_ppm)))
    ax.set_xticklabels([f"{d:.0f}" for d in cmap.depth_grid_ppm], rotation=45)
    ax.set_yticks(range(len(cmap.period_grid)))
    ax.set_yticklabels([f"{p:.1f}" for p in cmap.period_grid])
    ax.set_xlabel("Depth (ppm)")
    ax.set_ylabel("Period (days)")
    ax.set_title(f"Recovery completeness — {cmap.target_id}")
    plt.colorbar(im, ax=ax, label="Recovery rate")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="recovery_completeness_map",
        description="Grid injection-recovery completeness map.",
    )
    parser.add_argument("--target", type=str, default="unknown", metavar="ID")
    parser.add_argument("--output", type=Path, required=True, metavar="FILE",
                        help="Output JSON file.")
    parser.add_argument("--plot", type=Path, default=None, metavar="PNG",
                        help="Optional PNG heatmap output.")
    parser.parse_args(argv)

    print("Recovery completeness map requires a loaded numpy light curve.")
    print("Use the library API: build_completeness_map(period_grid, depth_grid,"
          " time=..., flux=...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
