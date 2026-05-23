"""Compute a 2-D period × depth detection efficiency map from injection-recovery.

Bins injection-recovery pairs into a period × depth grid and computes the
per-cell detection efficiency (fraction of injected signals recovered).
Complements ``recovery_completeness_map`` (overall totals) and
``injection_recovery`` (runs the injection itself).

Public API
----------
EfficiencyCell(period_lo, period_hi, depth_lo, depth_hi, n_injected,
               n_recovered, efficiency)
DetectionEfficiencyResult(n_cells, period_bins, depth_bins, cells, flag)
compute_detection_efficiency(injected, recovered, period_bins,
                             depth_bins) -> DetectionEfficiencyResult
format_efficiency_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficiencyCell:
    period_lo: float
    period_hi: float
    depth_lo: float
    depth_hi: float
    n_injected: int
    n_recovered: int
    efficiency: float | None  # None if n_injected == 0


@dataclass(frozen=True)
class DetectionEfficiencyResult:
    n_cells: int
    period_bins: tuple[float, ...]  # bin edges (n+1 values for n bins)
    depth_bins: tuple[float, ...]
    cells: tuple[EfficiencyCell, ...]
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _bin_index(value: float, edges: list[float]) -> int | None:
    """Return 0-based bin index for value given sorted bin edges."""
    if value < edges[0] or value > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= value <= edges[i + 1]:
            return i
    return None


def compute_detection_efficiency(
    injected: list[dict],
    recovered: list[dict],
    period_bins: list[float],
    depth_bins: list[float],
) -> DetectionEfficiencyResult:
    """Compute per-cell detection efficiency from injection-recovery data.

    Each dict in *injected* / *recovered* must contain ``period_days`` and
    ``depth_ppm`` keys.  A signal is matched as recovered if its ``period_days``
    and ``depth_ppm`` appear in *recovered* (exact match on rounded values).

    Args:
        injected: List of injected signal dicts (``period_days``, ``depth_ppm``).
        recovered: List of recovered signal dicts (same keys).
        period_bins: Bin edges for period axis (days).  Must have ≥ 2 edges.
        depth_bins: Bin edges for depth axis (ppm).  Must have ≥ 2 edges.

    Returns:
        :class:`DetectionEfficiencyResult`.
    """
    if not isinstance(injected, list) or not isinstance(recovered, list):
        return DetectionEfficiencyResult(0, (), (), (), "INVALID")
    if len(period_bins) < 2 or len(depth_bins) < 2:
        return DetectionEfficiencyResult(0, (), (), (), "INVALID")
    if not injected:
        return DetectionEfficiencyResult(
            0, tuple(period_bins), tuple(depth_bins), (), "INSUFFICIENT"
        )

    n_p = len(period_bins) - 1
    n_d = len(depth_bins) - 1

    # Count injected per cell
    grid_inj = [[0] * n_d for _ in range(n_p)]
    grid_rec = [[0] * n_d for _ in range(n_p)]

    # Build a set of recovered (rounded) for fast lookup
    rec_set: set[tuple[float, float]] = set()
    for r in recovered:
        p = r.get("period_days")
        d = r.get("depth_ppm")
        if p is not None and d is not None:
            rec_set.add((round(float(p), 4), round(float(d), 2)))

    for inj in injected:
        p = inj.get("period_days")
        d = inj.get("depth_ppm")
        if p is None or d is None:
            continue
        pi = _bin_index(float(p), period_bins)
        di = _bin_index(float(d), depth_bins)
        if pi is None or di is None:
            continue
        grid_inj[pi][di] += 1
        key = (round(float(p), 4), round(float(d), 2))
        if key in rec_set:
            grid_rec[pi][di] += 1

    cells: list[EfficiencyCell] = []
    for pi in range(n_p):
        for di in range(n_d):
            ni = grid_inj[pi][di]
            nr = grid_rec[pi][di]
            eff = round(nr / ni, 4) if ni > 0 else None
            cells.append(EfficiencyCell(
                period_lo=period_bins[pi],
                period_hi=period_bins[pi + 1],
                depth_lo=depth_bins[di],
                depth_hi=depth_bins[di + 1],
                n_injected=ni,
                n_recovered=nr,
                efficiency=eff,
            ))

    return DetectionEfficiencyResult(
        n_cells=len(cells),
        period_bins=tuple(period_bins),
        depth_bins=tuple(depth_bins),
        cells=tuple(cells),
        flag="OK",
    )


def format_efficiency_result(result: DetectionEfficiencyResult) -> str:
    """Format detection efficiency map as Markdown."""
    lines = [
        "## Detection Efficiency Map",
        "",
        f"- Period bins: {len(result.period_bins) - 1 if result.period_bins else 0}",
        f"- Depth bins: {len(result.depth_bins) - 1 if result.depth_bins else 0}",
        f"- Total cells: {result.n_cells}",
        f"- **Flag: {result.flag}**",
    ]
    filled = [c for c in result.cells if c.n_injected > 0]
    if filled:
        lines += ["", "Sample cells (period lo–hi d, depth lo–hi ppm, efficiency):"]
        for c in filled[:5]:
            eff_s = f"{c.efficiency:.3f}" if c.efficiency is not None else "—"
            lines.append(
                f"  - P=[{c.period_lo}–{c.period_hi}], D=[{c.depth_lo}–{c.depth_hi}]:"
                f" {c.n_recovered}/{c.n_injected} = {eff_s}"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="detection_efficiency_map",
        description="Compute 2-D detection efficiency from injection-recovery data.",
    )
    parser.parse_args(argv)

    result = compute_detection_efficiency([], [], [1, 5, 20], [100, 1000, 5000])
    print(format_efficiency_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
