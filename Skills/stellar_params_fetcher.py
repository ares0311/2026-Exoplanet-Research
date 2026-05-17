"""Fetch stellar parameters from the TESS Input Catalog (TIC) via astroquery.

Returns a StellarParams dataclass whose to_vet_kwargs() method produces a dict
compatible with vet_signal() keyword arguments.

Public API
----------
fetch_stellar_params(tic_id, *, catalog_fn) -> StellarParams
StellarParams.to_vet_kwargs() -> dict
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class StellarParams:
    tic_id: int
    stellar_radius_rsun: float | None
    stellar_mass_msun: float | None
    stellar_teff_k: float | None
    stellar_logg: float | None
    contamination_ratio: float | None

    def to_vet_kwargs(self) -> dict[str, Any]:
        """Return non-None fields as kwargs for ``vet_signal()``.

        Excludes ``tic_id`` and ``stellar_logg`` which are not accepted by
        ``vet_signal``; includes ``stellar_radius_rsun``, ``stellar_mass_msun``,
        ``stellar_teff_k``, ``contamination_ratio``.
        """
        mapping = {
            "stellar_radius_rsun": self.stellar_radius_rsun,
            "stellar_mass_msun": self.stellar_mass_msun,
            "stellar_teff_k": self.stellar_teff_k,
            "contamination_ratio": self.contamination_ratio,
        }
        return {k: v for k, v in mapping.items() if v is not None}


def _safe_float(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _default_catalog_fn(tic_id: int) -> dict[str, Any]:
    """Query TIC via astroquery (live network call)."""
    from astroquery.mast import Catalogs  # noqa: PLC0415
    results = Catalogs.query_criteria("TIC", ID=tic_id, objType="STAR")
    if len(results) == 0:
        return {}
    row = results[0]
    return {
        "rad": row["rad"],
        "mass": row["mass"],
        "Teff": row["Teff"],
        "logg": row["logg"],
        "contratio": row["contratio"],
    }


def fetch_stellar_params(
    tic_id: int,
    *,
    catalog_fn: Callable[[int], dict[str, Any]] | None = None,
) -> StellarParams:
    """Fetch stellar parameters for a TIC target.

    Args:
        tic_id: TESS Input Catalog identifier.
        catalog_fn: Injectable ``(tic_id) -> dict`` returning TIC column values.
            Keys expected: ``rad``, ``mass``, ``Teff``, ``logg``, ``contratio``.
            Defaults to a live astroquery call.

    Returns:
        :class:`StellarParams` dataclass.
    """
    fn = catalog_fn if catalog_fn is not None else _default_catalog_fn
    row = fn(tic_id)
    return StellarParams(
        tic_id=tic_id,
        stellar_radius_rsun=_safe_float(row, "rad"),
        stellar_mass_msun=_safe_float(row, "mass"),
        stellar_teff_k=_safe_float(row, "Teff"),
        stellar_logg=_safe_float(row, "logg"),
        contamination_ratio=_safe_float(row, "contratio"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="stellar_params_fetcher",
        description="Fetch stellar parameters from the TIC for a target.",
    )
    parser.add_argument("tic_id", type=int, metavar="TIC_ID",
                        help="TESS Input Catalog ID.")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of a table.")
    args = parser.parse_args(argv)

    params = fetch_stellar_params(args.tic_id)
    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(params), indent=2))
    else:
        print(f"TIC {params.tic_id}")
        print(f"  Radius (R☉):  {params.stellar_radius_rsun}")
        print(f"  Mass (M☉):    {params.stellar_mass_msun}")
        print(f"  Teff (K):     {params.stellar_teff_k}")
        print(f"  log g:        {params.stellar_logg}")
        print(f"  Contamination:{params.contamination_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
