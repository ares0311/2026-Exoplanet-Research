"""Fetch comprehensive TIC metadata for a target star.

Combines TIC catalog data (position, magnitudes, stellar params) into a
single structured object for use as pipeline inputs.

Public API
----------
TargetMetadata(tic_id, ra, dec, tmag, teff, logg, radius_rsun, mass_msun,
               distance_pc, contratio, n_sectors, source)
fetch_target_metadata(tic_id, *, catalog_fn) -> TargetMetadata
format_target_metadata(meta) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetMetadata:
    tic_id: int
    ra: float | None
    dec: float | None
    tmag: float | None
    teff: float | None              # K
    logg: float | None              # log(g) in cgs
    radius_rsun: float | None
    mass_msun: float | None
    distance_pc: float | None
    contratio: float | None         # contamination ratio (neighbour flux / target)
    n_sectors: int                  # number of TESS sectors with data
    source: str                     # "TIC" or "stub"


def _default_catalog_fn(tic_id: int) -> dict:
    """Stub — returns empty dict; real impl uses astroquery.mast.Catalogs."""
    return {}


def fetch_target_metadata(
    tic_id: int,
    *,
    catalog_fn=None,
) -> TargetMetadata:
    """Fetch TIC metadata for a target.

    Args:
        tic_id: TESS Input Catalog identifier.
        catalog_fn: Injectable; called as ``catalog_fn(tic_id) -> dict``
            returning TIC column values.  Keys accepted:
            ``ra``, ``dec``, ``Tmag``, ``Teff``, ``logg``, ``rad``, ``mass``,
            ``d``, ``contratio``, ``numcont``.

    Returns:
        :class:`TargetMetadata`.
    """
    if catalog_fn is None:
        catalog_fn = _default_catalog_fn

    row = catalog_fn(tic_id) or {}

    def _f(key: str) -> float | None:
        val = row.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return None if (f != f) else f  # NaN check
        except (TypeError, ValueError):
            return None

    def _i(key: str, default: int = 0) -> int:
        val = row.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    source = "TIC" if row else "stub"

    return TargetMetadata(
        tic_id=tic_id,
        ra=_f("ra"),
        dec=_f("dec"),
        tmag=_f("Tmag"),
        teff=_f("Teff"),
        logg=_f("logg"),
        radius_rsun=_f("rad"),
        mass_msun=_f("mass"),
        distance_pc=_f("d"),
        contratio=_f("contratio"),
        n_sectors=_i("n_sectors"),
        source=source,
    )


def format_target_metadata(meta: TargetMetadata) -> str:
    """Format target metadata as Markdown."""
    def _fmt(val: float | None, unit: str = "", fmt: str = ".2f") -> str:
        if val is None:
            return "—"
        return f"{val:{fmt}} {unit}".strip()

    lines = [
        f"## Target Metadata — TIC {meta.tic_id}",
        f"_(source: {meta.source})_",
        "",
        f"- RA / Dec: {_fmt(meta.ra)} / {_fmt(meta.dec)} deg",
        f"- Tmag: {_fmt(meta.tmag)}",
        f"- Teff: {_fmt(meta.teff, 'K', '.0f')}",
        f"- log g: {_fmt(meta.logg)}",
        f"- Radius: {_fmt(meta.radius_rsun, 'R☉')}",
        f"- Mass: {_fmt(meta.mass_msun, 'M☉')}",
        f"- Distance: {_fmt(meta.distance_pc, 'pc', '.0f')}",
        f"- Contamination ratio: {_fmt(meta.contratio, fmt='.4f')}",
        f"- TESS sectors: {meta.n_sectors}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="target_metadata_fetcher",
        description="Fetch comprehensive TIC metadata for a target star.",
    )
    parser.add_argument("tic_id", type=int)
    args = parser.parse_args(argv)

    meta = fetch_target_metadata(args.tic_id)
    print(format_target_metadata(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
