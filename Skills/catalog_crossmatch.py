"""Cross-match a TIC ID against external catalogs (Simbad, Gaia, 2MASS, NEA).

Provides injectable catalog-query functions so all network calls can be mocked
in tests.  Results are aggregated into a CatalogMatch dataclass.

Public API
----------
CatalogMatch(tic_id, simbad_name, gaia_source_id, twomass_id,
             nea_planet_name, distance_pc, spectral_type, found_in)
crossmatch(tic_id, ra, dec, *, simbad_fn, gaia_fn, nea_fn) -> CatalogMatch
format_crossmatch(match) -> str
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CatalogMatch:
    tic_id: int
    ra: float
    dec: float
    simbad_name: str | None         # Main identifier from Simbad
    gaia_source_id: str | None      # Gaia DR3 source_id
    twomass_id: str | None          # 2MASS designation
    nea_planet_name: str | None     # NEA confirmed planet name (if any)
    distance_pc: float | None       # Gaia parallax-derived distance
    spectral_type: str | None       # Simbad spectral type
    found_in: tuple[str, ...]       # which catalogs returned a hit


# ---------------------------------------------------------------------------
# Default catalog query stubs (require network; injectable in tests)
# ---------------------------------------------------------------------------

def _simbad_query(ra: float, dec: float, radius_arcsec: float = 5.0) -> dict[str, Any]:
    """Query Simbad by position; return dict with main_id, sp_type."""
    from astroquery.simbad import Simbad  # type: ignore[import]
    Simbad.add_votable_fields("sp_type", "distance")
    result = Simbad.query_region(
        f"{ra} {dec}",
        radius=f"{radius_arcsec}s",
    )
    if result is None or len(result) == 0:
        return {}
    row = result[0]
    return {
        "main_id": str(row["MAIN_ID"]),
        "sp_type": str(row.get("SP_TYPE", "")) or None,
    }


def _gaia_query(ra: float, dec: float, radius_arcsec: float = 5.0) -> dict[str, Any]:
    """Query Gaia DR3 by position; return source_id and parallax distance."""

    from astroquery.gaia import Gaia  # type: ignore[import]
    radius_deg = radius_arcsec / 3600.0
    job = Gaia.cone_search_async(
        ra=ra, dec=dec, radius=radius_deg,
        table_name="gaiadr3.gaia_source",
        columns=["source_id", "parallax"],
    )
    result = job.get_results()
    if result is None or len(result) == 0:
        return {}
    row = result[0]
    parallax = float(row["parallax"]) if row["parallax"] else None
    dist_pc = 1000.0 / parallax if parallax and parallax > 0 else None
    return {
        "source_id": str(row["source_id"]),
        "distance_pc": dist_pc,
    }


def _nea_query(tic_id: int) -> dict[str, Any]:
    """Check NASA Exoplanet Archive for confirmed planets around TIC target."""
    import json as _json
    import urllib.request
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        f"select+pl_name,tic_id+from+ps+where+tic_id={tic_id}"
        "&format=json"
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        data = _json.loads(r.read())
    if data:
        return {"planet_name": data[0]["pl_name"]}
    return {}


# ---------------------------------------------------------------------------
# Public crossmatch function
# ---------------------------------------------------------------------------

def crossmatch(
    tic_id: int,
    ra: float,
    dec: float,
    *,
    simbad_fn: Callable[[float, float], dict] | None = None,
    gaia_fn: Callable[[float, float], dict] | None = None,
    nea_fn: Callable[[int], dict] | None = None,
    radius_arcsec: float = 5.0,
) -> CatalogMatch:
    """Cross-match a TIC target against Simbad, Gaia DR3, and NEA.

    Args:
        tic_id: TESS Input Catalog ID.
        ra: Right ascension (degrees, J2000).
        dec: Declination (degrees, J2000).
        simbad_fn: Injectable ``(ra, dec) -> dict``.
        gaia_fn: Injectable ``(ra, dec) -> dict``.
        nea_fn: Injectable ``(tic_id) -> dict``.
        radius_arcsec: Search cone radius in arcseconds.

    Returns:
        :class:`CatalogMatch`.
    """
    _sim = simbad_fn if simbad_fn is not None else (
        lambda ra_, dec_: _simbad_query(ra_, dec_, radius_arcsec)
    )
    _gaia = gaia_fn if gaia_fn is not None else (
        lambda ra_, dec_: _gaia_query(ra_, dec_, radius_arcsec)
    )
    _nea = nea_fn if nea_fn is not None else _nea_query

    found_in: list[str] = []

    sim = {}
    try:
        sim = _sim(ra, dec)
        if sim:
            found_in.append("simbad")
    except Exception:
        pass

    gaia: dict[str, Any] = {}
    try:
        gaia = _gaia(ra, dec)
        if gaia:
            found_in.append("gaia")
    except Exception:
        pass

    nea: dict[str, Any] = {}
    try:
        nea = _nea(tic_id)
        if nea:
            found_in.append("nea")
    except Exception:
        pass

    # Build 2MASS ID heuristically from Simbad main_id (often contains "2MASS J…")
    twomass: str | None = None
    main_id = sim.get("main_id", "")
    if main_id and "2MASS" in main_id:
        twomass = main_id

    return CatalogMatch(
        tic_id=tic_id,
        ra=ra,
        dec=dec,
        simbad_name=sim.get("main_id") or None,
        gaia_source_id=gaia.get("source_id") or None,
        twomass_id=twomass,
        nea_planet_name=nea.get("planet_name") or None,
        distance_pc=gaia.get("distance_pc"),
        spectral_type=sim.get("sp_type") or None,
        found_in=tuple(found_in),
    )


def format_crossmatch(match: CatalogMatch) -> str:
    """Format a CatalogMatch as a Markdown block."""
    lines = [
        "## Catalog Crossmatch",
        "",
        f"- TIC ID: {match.tic_id}",
        f"- RA / Dec: {match.ra:.5f}° / {match.dec:.5f}°",
        f"- Simbad: {match.simbad_name or '—'}",
        f"- Spectral type: {match.spectral_type or '—'}",
        f"- Gaia DR3 source: {match.gaia_source_id or '—'}",
        f"- Distance: {f'{match.distance_pc:.0f} pc' if match.distance_pc else '—'}",
        f"- 2MASS: {match.twomass_id or '—'}",
        f"- NEA planet: {match.nea_planet_name or 'none found'}",
        f"- Found in: {', '.join(match.found_in) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="catalog_crossmatch",
        description="Cross-match a TIC target against Simbad, Gaia, and NEA.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--radius", type=float, default=5.0, metavar="ARCSEC")
    args = parser.parse_args(argv)

    match = crossmatch(args.tic_id, args.ra, args.dec, radius_arcsec=args.radius)
    print(format_crossmatch(match))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
