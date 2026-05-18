"""Compute quadratic limb-darkening coefficients from stellar Teff and logg.

Uses a hardcoded grid of Claret (2011) TESS-band coefficients interpolated
bilinearly in Teff and logg space.

Public API
----------
LimbDarkeningResult(teff_k, logg, u1, u2, gamma, passband, flag)
compute_limb_darkening(teff_k, logg, *, passband) -> LimbDarkeningResult
format_ld_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# Claret (2011) TESS-band grid: (Teff, logg) -> (u1, u2)
# Rows: logg = 3.5, 4.0, 4.5, 5.0
# Cols: Teff = 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000
_TEFF_GRID = [3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0]
_LOGG_GRID = [3.5, 4.0, 4.5, 5.0]

# u1 coefficients
_U1 = [
    [0.587, 0.540, 0.497, 0.451, 0.413, 0.373, 0.344, 0.318],
    [0.608, 0.558, 0.508, 0.461, 0.422, 0.381, 0.349, 0.320],
    [0.624, 0.574, 0.521, 0.472, 0.430, 0.389, 0.357, 0.326],
    [0.637, 0.588, 0.534, 0.483, 0.439, 0.397, 0.363, 0.332],
]
# u2 coefficients
_U2 = [
    [0.162, 0.175, 0.188, 0.198, 0.208, 0.213, 0.217, 0.219],
    [0.147, 0.161, 0.175, 0.186, 0.197, 0.204, 0.209, 0.213],
    [0.133, 0.148, 0.163, 0.175, 0.187, 0.195, 0.201, 0.207],
    [0.120, 0.136, 0.152, 0.165, 0.178, 0.187, 0.194, 0.201],
]


@dataclass(frozen=True)
class LimbDarkeningResult:
    teff_k: float
    logg: float
    u1: float
    u2: float
    gamma: float           # 1 - u1/3 - u2/6
    passband: str
    flag: str              # "OK", "EXTRAPOLATED", "INVALID"


def _bilinear(grid_x: list[float], grid_y: list[float],
               values: list[list[float]], x: float, y: float) -> tuple[float, bool]:
    """Bilinear interpolation; returns (value, extrapolated_flag)."""
    x_lo = max(grid_x[0], min(grid_x[-1], x))
    y_lo = max(grid_y[0], min(grid_y[-1], y))
    extrapolated = (x != x_lo or y != y_lo)

    # Find bracketing indices for x
    ix = 0
    for i in range(len(grid_x) - 1):
        if grid_x[i] <= x_lo <= grid_x[i + 1]:
            ix = i
            break
    else:
        ix = len(grid_x) - 2

    # Find bracketing indices for y
    iy = 0
    for i in range(len(grid_y) - 1):
        if grid_y[i] <= y_lo <= grid_y[i + 1]:
            iy = i
            break
    else:
        iy = len(grid_y) - 2

    x0, x1 = grid_x[ix], grid_x[ix + 1]
    y0, y1 = grid_y[iy], grid_y[iy + 1]
    dx = (x_lo - x0) / (x1 - x0) if x1 != x0 else 0.0
    dy = (y_lo - y0) / (y1 - y0) if y1 != y0 else 0.0

    v00 = values[iy][ix]
    v10 = values[iy][ix + 1]
    v01 = values[iy + 1][ix]
    v11 = values[iy + 1][ix + 1]
    val = (v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy)
           + v01 * (1 - dx) * dy + v11 * dx * dy)
    return val, extrapolated


def compute_limb_darkening(
    teff_k: float,
    logg: float,
    *,
    passband: str = "TESS",
) -> LimbDarkeningResult:
    """Compute quadratic limb-darkening coefficients.

    Args:
        teff_k: Stellar effective temperature in Kelvin.
        logg: Stellar surface gravity (log g in cgs).
        passband: Photometric passband (only "TESS" supported).

    Returns:
        :class:`LimbDarkeningResult`.
    """
    if teff_k <= 0 or logg < 0:
        return LimbDarkeningResult(teff_k, logg, 0.0, 0.0, 1.0, passband, "INVALID")

    u1, ext1 = _bilinear(_TEFF_GRID, _LOGG_GRID, _U1, float(teff_k), float(logg))
    u2, ext2 = _bilinear(_TEFF_GRID, _LOGG_GRID, _U2, float(teff_k), float(logg))
    extrapolated = ext1 or ext2

    gamma = 1.0 - u1 / 3.0 - u2 / 6.0

    return LimbDarkeningResult(
        teff_k=float(teff_k),
        logg=float(logg),
        u1=round(u1, 4),
        u2=round(u2, 4),
        gamma=round(gamma, 4),
        passband=passband,
        flag="EXTRAPOLATED" if extrapolated else "OK",
    )


def format_ld_result(result: LimbDarkeningResult) -> str:
    """Format limb-darkening result as Markdown."""
    lines = [
        "## Limb Darkening Coefficients",
        "",
        f"- Teff: {result.teff_k:.0f} K",
        f"- log g: {result.logg:.2f}",
        f"- Passband: {result.passband}",
        f"- u1 = {result.u1:.4f}",
        f"- u2 = {result.u2:.4f}",
        f"- γ (transit shape) = {result.gamma:.4f}",
        f"- Flag: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="limb_darkening_calculator",
        description="Compute quadratic limb-darkening coefficients.",
    )
    parser.add_argument("teff_k", type=float)
    parser.add_argument("logg", type=float)
    parser.add_argument("--passband", default="TESS")
    args = parser.parse_args(argv)

    result = compute_limb_darkening(args.teff_k, args.logg, passband=args.passband)
    print(format_ld_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
