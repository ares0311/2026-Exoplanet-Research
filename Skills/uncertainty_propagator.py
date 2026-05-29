"""Propagate parameter uncertainties through a user-supplied function.

Uses symmetric finite differences to compute partial derivatives and
combines them in quadrature (linear error propagation).

Public API
----------
PropagationResult(output_value, output_uncertainty, contributions,
                  relative_uncertainty, flag)
propagate_uncertainty(func, params, uncertainties, *,
                      step_fraction) -> PropagationResult
format_propagation_result(result) -> str
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationResult:
    output_value: float
    output_uncertainty: float
    contributions: tuple[tuple[str, float], ...]  # (param_name, abs_contribution)
    relative_uncertainty: float | None
    flag: str  # "OK" | "LARGE_UNCERTAINTY" | "INVALID"


def propagate_uncertainty(
    func: Callable[..., float],
    params: dict[str, float],
    uncertainties: dict[str, float],
    *,
    step_fraction: float = 1e-4,
) -> PropagationResult:
    """Propagate parameter uncertainties via finite differences.

    Args:
        func: Function accepting keyword arguments matching params keys.
        params: Central-value parameter dict.
        uncertainties: 1-sigma uncertainties for each parameter.
        step_fraction: Step size as fraction of each parameter value.

    Returns:
        PropagationResult with combined uncertainty and per-param contributions.
    """
    try:
        central = func(**params)
    except Exception:
        return PropagationResult(
            output_value=float("nan"),
            output_uncertainty=float("nan"),
            contributions=(),
            relative_uncertainty=None,
            flag="INVALID",
        )

    if not math.isfinite(central):
        return PropagationResult(
            output_value=central,
            output_uncertainty=float("nan"),
            contributions=(),
            relative_uncertainty=None,
            flag="INVALID",
        )

    contributions: list[tuple[str, float]] = []
    variance = 0.0

    for name, sigma in uncertainties.items():
        if name not in params or sigma <= 0:
            continue
        val = params[name]
        h = abs(val) * step_fraction if abs(val) > 1e-12 else step_fraction

        p_plus = {**params, name: val + h}
        p_minus = {**params, name: val - h}
        try:
            f_plus = func(**p_plus)
            f_minus = func(**p_minus)
        except Exception:
            continue

        if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
            continue

        deriv = (f_plus - f_minus) / (2 * h)
        contrib = abs(deriv * sigma)
        variance += contrib ** 2
        contributions.append((name, round(contrib, 6)))

    contributions.sort(key=lambda x: x[1], reverse=True)
    sigma_out = math.sqrt(variance)

    rel = abs(sigma_out / central) if abs(central) > 1e-12 else None
    flag = "LARGE_UNCERTAINTY" if (rel is not None and rel > 0.50) else "OK"

    return PropagationResult(
        output_value=round(central, 6),
        output_uncertainty=round(sigma_out, 6),
        contributions=tuple(contributions),
        relative_uncertainty=round(rel, 4) if rel is not None else None,
        flag=flag,
    )


def format_propagation_result(result: PropagationResult) -> str:
    """Format propagation result as Markdown.

    Args:
        result: PropagationResult to format.

    Returns:
        Markdown string.
    """
    rel_str = (f"{result.relative_uncertainty * 100:.1f}%"
               if result.relative_uncertainty is not None else "—")
    lines = [
        "## Uncertainty Propagation\n",
        f"**Status**: `{result.flag}` | "
        f"Result: {result.output_value:.6g} ± {result.output_uncertainty:.6g} "
        f"({rel_str})\n",
    ]
    if not result.contributions:
        lines.append("\n_No contributions computed._")
        return "\n".join(lines)

    lines += [
        "",
        "| Parameter | Contribution (1σ) |",
        "|---|---|",
    ]
    total = result.output_uncertainty
    for name, contrib in result.contributions:
        pct = f"{100 * contrib / total:.1f}%" if total > 0 else "—"
        lines.append(f"| {name} | {contrib:.6g} ({pct}) |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Propagate parameter uncertainties.")
    parser.add_argument(
        "params_file",
        help="JSON file with {'params': {...}, 'uncertainties': {...}}.",
    )
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.params_file).read_text())

    # Simple demonstration: product of all params
    def _product(**kwargs: float) -> float:
        result = 1.0
        for v in kwargs.values():
            result *= v
        return result

    result = propagate_uncertainty(
        _product,
        params=data["params"],
        uncertainties=data["uncertainties"],
    )
    print(format_propagation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
