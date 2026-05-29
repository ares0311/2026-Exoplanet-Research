"""Compute quadratic limb darkening profile.

I(mu)/I(0) = 1 - u1*(1-mu) - u2*(1-mu)^2

Public API
----------
LimbProfileResult(mu_values, intensity_values, u1, u2, flag)
compute_limb_profile(u1, u2, n_points) -> LimbProfileResult
format_limb_profile(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LimbProfileResult:
    mu_values: tuple[float, ...]
    intensity_values: tuple[float, ...]
    u1: float
    u2: float
    flag: str  # "OK" or "UNPHYSICAL" if any intensity < 0


def compute_limb_profile(
    u1: float = 0.4,
    u2: float = 0.3,
    n_points: int = 10,
) -> LimbProfileResult:
    """Compute quadratic limb darkening profile.

    Args:
        u1: First limb darkening coefficient.
        u2: Second limb darkening coefficient.
        n_points: Number of sample points from mu=0 to mu=1 (inclusive).

    Returns:
        :class:`LimbProfileResult`.
    """
    if n_points < 2:
        n_points = 2

    step = 1.0 / (n_points - 1)
    mu_vals: list[float] = []
    int_vals: list[float] = []
    unphysical = False

    for i in range(n_points):
        mu = i * step
        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        mu_vals.append(round(mu, 8))
        int_vals.append(round(intensity, 8))
        if intensity < 0.0:
            unphysical = True

    flag = "UNPHYSICAL" if unphysical else "OK"
    return LimbProfileResult(
        mu_values=tuple(mu_vals),
        intensity_values=tuple(int_vals),
        u1=u1,
        u2=u2,
        flag=flag,
    )


def format_limb_profile(result: LimbProfileResult) -> str:
    """Format limb profile as a plain-text table."""
    lines = [
        "## Stellar Limb Profile (Quadratic)",
        "",
        f"- u1 = {result.u1}, u2 = {result.u2}",
        f"- Flag: {result.flag}",
        "",
        "| mu   | I(mu)/I(0) |",
        "|------|------------|",
    ]
    for mu, intensity in zip(result.mu_values, result.intensity_values):
        lines.append(f"| {mu:.3f} | {intensity:.6f}   |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="stellar_limb_profile",
        description=__doc__,
    )
    p.add_argument("--u1", type=float, default=0.4, help="First LD coefficient")
    p.add_argument("--u2", type=float, default=0.3, help="Second LD coefficient")
    p.add_argument("--n-points", type=int, default=10, help="Number of sample points")
    args = p.parse_args(argv)
    r = compute_limb_profile(args.u1, args.u2, args.n_points)
    print(format_limb_profile(r))
    return 0 if r.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
