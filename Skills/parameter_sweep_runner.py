"""Generic grid-search over a callable across a parameter space.

Accepts a user-supplied callable and a ``param_grid`` dict mapping parameter
names to lists of candidate values.  Evaluates the callable at every point in
the Cartesian product of the grid and returns the best combination together
with all sweep results.

Public API
----------
SweepPoint(params, metric_value)
SweepResult(n_points, best_params, best_value, sweep_points, flag)
run_parameter_sweep(fn, param_grid, *, maximize) -> SweepResult
format_sweep_result(result) -> str
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class SweepPoint:
    params: dict   # parameter name → value (frozen via tuple repr)
    metric_value: float | None


@dataclass(frozen=True)
class SweepResult:
    n_points: int
    best_params: dict | None
    best_value: float | None
    sweep_points: tuple[SweepPoint, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def run_parameter_sweep(
    fn,
    param_grid: dict[str, list],
    *,
    maximize: bool = True,
) -> SweepResult:
    """Evaluate *fn* at every point in the Cartesian product of *param_grid*.

    Args:
        fn: Callable that accepts ``**kwargs`` and returns a scalar metric.
        param_grid: Dict mapping parameter names to lists of candidate values.
        maximize: If True (default), select the point with the highest metric.
            If False, select the minimum.

    Returns:
        :class:`SweepResult`.
    """
    if not callable(fn):
        return SweepResult(0, None, None, (), "INVALID")
    if not param_grid:
        return SweepResult(0, None, None, (), "EMPTY")

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    if any(not isinstance(v, (list, tuple)) or len(v) == 0 for v in values):
        return SweepResult(0, None, None, (), "INVALID")

    sweep_points: list[SweepPoint] = []
    best_params: dict | None = None
    best_value: float | None = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo, strict=False))
        try:
            metric = float(fn(**params))
        except Exception:
            metric = None

        sweep_points.append(SweepPoint(params=params, metric_value=metric))

        if metric is not None and (
            best_value is None
            or (maximize and metric > best_value)
            or (not maximize and metric < best_value)
        ):
            best_value = metric
            best_params = params

    return SweepResult(
        n_points=len(sweep_points),
        best_params=best_params,
        best_value=round(best_value, 8) if best_value is not None else None,
        sweep_points=tuple(sweep_points),
        flag="OK",
    )


def format_sweep_result(result: SweepResult) -> str:
    """Format parameter sweep result as Markdown."""
    lines = [
        "## Parameter Sweep Runner",
        "",
        f"- Grid points evaluated: {result.n_points}",
        f"- **Best metric value: {result.best_value}**",
    ]
    if result.best_params:
        lines.append("- **Best parameters:**")
        for k, v in result.best_params.items():
            lines.append(f"  - `{k}` = {v}")
    lines.append(f"- **Flag: {result.flag}**")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="parameter_sweep_runner",
        description="Grid-sweep a callable over a parameter space.",
    )
    parser.parse_args(argv)

    def _demo(x=1, y=1):
        return -(x - 2) ** 2 - (y - 3) ** 2

    result = run_parameter_sweep(_demo, {"x": [1, 2, 3], "y": [2, 3, 4]})
    print(format_sweep_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
