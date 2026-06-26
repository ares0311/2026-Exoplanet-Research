"""Query MAST for JWST time-series observations suitable for transit search.

Searches the MAST archive for all JWST observations with dataproduct_type=timeseries
and returns a structured list of available targets with their instrument, program,
and observation metadata.

This is the first step in Option A of the discovery pipeline: identify which JWST
targets have time-series photometry available before downloading data.

Output JSON schema (one record per observation)::

    {
        "obsid": "87654321",
        "target_name": "WASP-39",
        "ra": 185.123,
        "dec": -3.444,
        "instrument": "NIRISS/SOSS",
        "program_id": "1366",
        "t_min": 59800.0,          # MJD start
        "t_max": 59801.2,          # MJD end
        "t_exptime": 4320.0,       # seconds
        "n_products": 12,          # count of data products available
        "has_calints": true,       # _calints.fits available (Stage 2)
        "has_x1dints": false,      # _x1dints.fits available (NIRISS spectra)
        "dataURL": "mast:JWST/..."
    }

Usage::

    .venv/bin/python Skills/fetch_jwst_targets.py \\
        --output data/jwst_timeseries_targets.json \\
        [--instrument "NIRISS/SOSS"] \\
        [--min-exptime 1800] \\
        [--json]

    # JSON mode prints the list to stdout for piping
    .venv/bin/python Skills/fetch_jwst_targets.py --json | python -m json.tool | head -80

Public API
----------
JwstObservation   dataclass for one MAST observation record
query_jwst_timeseries(instrument, min_exptime, search_fn) -> list[JwstObservation]
format_summary(obs_list) -> str
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JwstObservation:
    obsid: str
    target_name: str
    ra: float | None
    dec: float | None
    instrument: str
    program_id: str
    t_min: float | None          # MJD
    t_max: float | None          # MJD
    t_exptime: float | None      # total seconds
    n_products: int
    has_calints: bool            # Stage 2 calibrated integrations
    has_x1dints: bool            # Stage 3 NIRISS extracted 1D spectra
    filters: str


# ---------------------------------------------------------------------------
# MAST query helpers
# ---------------------------------------------------------------------------

_INSTRUMENTS_TIMESERIES = [
    "NIRISS/SOSS",       # Near Infrared Imager and Slitless Spectrograph, SOSS mode
    "NIRCAM/GRISM TIME SERIES",  # NIRCam grism time-series
    "NIRCAM/TIME SERIES IMAGING", # NIRCam photometric time-series
    "MIRI/TIME SERIES IMAGING",  # MIRI photometric time-series
    "NIRSPEC/BOTS",      # NIRSpec Bright Object Time Series
]


def _default_search(instrument_name: str | None, min_exptime: float) -> list[dict[str, Any]]:
    """Query MAST for JWST time-series observations via astroquery."""
    from astroquery.mast import Observations  # type: ignore[import]

    constraints: dict[str, Any] = {
        "obs_collection": "JWST",
        "dataproduct_type": "timeseries",
    }
    if instrument_name:
        constraints["instrument_name"] = instrument_name

    table = Observations.query_criteria(**constraints)
    if table is None or len(table) == 0:
        return []

    rows = []
    for row in table:
        exptime = float(row.get("t_exptime", 0) or 0)
        if exptime < min_exptime:
            continue
        rows.append({
            "obsid": str(row.get("obsid", "")),
            "target_name": str(row.get("target_name", "")),
            "ra": _safe_float(row.get("s_ra")),
            "dec": _safe_float(row.get("s_dec")),
            "instrument": str(row.get("instrument_name", "")),
            "program_id": str(row.get("proposal_id", "")),
            "t_min": _safe_float(row.get("t_min")),
            "t_max": _safe_float(row.get("t_max")),
            "t_exptime": exptime,
            "filters": str(row.get("filters", "")),
            "obsid_raw": str(row.get("obsid", "")),
        })
    return rows


def _enrich_with_products(
    raw_rows: list[dict[str, Any]],
    product_fn: Callable[[list[str]], dict[str, list[str]]] | None = None,
) -> list[JwstObservation]:
    """Fetch product lists and annotate each observation with availability flags."""
    if not raw_rows:
        return []

    if product_fn is None:
        product_fn = _default_product_fn

    obsids = [r["obsid_raw"] for r in raw_rows]
    product_map = product_fn(obsids)

    result = []
    for row in raw_rows:
        oid = row["obsid_raw"]
        products = product_map.get(oid, [])
        has_calints = any("calints" in p.lower() for p in products)
        has_x1dints = any("x1dints" in p.lower() for p in products)
        result.append(JwstObservation(
            obsid=row["obsid"],
            target_name=row["target_name"],
            ra=row["ra"],
            dec=row["dec"],
            instrument=row["instrument"],
            program_id=row["program_id"],
            t_min=row["t_min"],
            t_max=row["t_max"],
            t_exptime=row["t_exptime"],
            n_products=len(products),
            has_calints=has_calints,
            has_x1dints=has_x1dints,
            filters=row["filters"],
        ))
    return result


def _default_product_fn(obsids: list[str]) -> dict[str, list[str]]:
    """Return {obsid: [filename, ...]} for each observation."""
    from astroquery.mast import Observations  # type: ignore[import]

    product_map: dict[str, list[str]] = {oid: [] for oid in obsids}
    if not obsids:
        return product_map
    # Batch in groups of 50 to stay polite
    batch_size = 50
    for i in range(0, len(obsids), batch_size):
        batch = obsids[i : i + batch_size]
        try:
            products = Observations.get_product_list(batch)
            for row in products:
                oid = str(row.get("parent_obsid", ""))
                fname = str(row.get("productFilename", ""))
                if oid in product_map:
                    product_map[oid].append(fname)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    return product_map


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if (f != f) else f  # nan check
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def query_jwst_timeseries(
    instrument: str | None = None,
    min_exptime: float = 1800.0,
    search_fn: Callable[[str | None, float], list[dict[str, Any]]] | None = None,
    product_fn: Callable[[list[str]], dict[str, list[str]]] | None = None,
) -> list[JwstObservation]:
    """Query MAST for JWST time-series observations.

    Args:
        instrument: JWST instrument name filter (None = all time-series modes).
        min_exptime: Minimum total exposure time in seconds (default 1800 = 30 min).
        search_fn: Injectable MAST query function for testing.
        product_fn: Injectable product-list function for testing.

    Returns:
        List of JwstObservation records sorted by t_min ascending.
    """
    fn = search_fn or _default_search
    raw = fn(instrument, min_exptime)
    obs = _enrich_with_products(raw, product_fn)
    obs.sort(key=lambda o: (o.t_min or 0.0))
    return obs


def format_summary(obs_list: list[JwstObservation]) -> str:
    """Return a human-readable Markdown table of observations."""
    if not obs_list:
        return "_No JWST time-series observations found._"

    header = "| Target | Instrument | Program | Exptime (h) | calints | x1dints |"
    divider = "|--------|-----------|---------|-------------|---------|---------|"
    rows = []
    for o in obs_list:
        exp_h = f"{(o.t_exptime or 0) / 3600:.1f}"
        rows.append(
            f"| {o.target_name} | {o.instrument} | {o.program_id} "
            f"| {exp_h} | {'✓' if o.has_calints else '—'} | {'✓' if o.has_x1dints else '—'} |"
        )
    return "\n".join([header, divider] + rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="List JWST time-series observations available on MAST"
    )
    parser.add_argument("--output", help="Write JSON list to this path")
    parser.add_argument("--instrument", default=None, help="Filter by instrument name")
    parser.add_argument(
        "--min-exptime", type=float, default=1800.0,
        help="Minimum total exposure time in seconds (default: 1800)"
    )
    parser.add_argument("--json", action="store_true", dest="json_mode",
                        help="Print JSON to stdout instead of summary table")
    args = parser.parse_args()

    print("Querying MAST for JWST time-series observations...", flush=True)
    start = time.monotonic()

    try:
        obs = query_jwst_timeseries(instrument=args.instrument, min_exptime=args.min_exptime)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.monotonic() - start
    print(f"Found {len(obs)} observations in {elapsed:.1f}s", flush=True)

    if args.json_mode:
        print(json.dumps([asdict(o) for o in obs], indent=2))
    else:
        print(format_summary(obs))

    if args.output:
        import pathlib
        p = pathlib.Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(o) for o in obs], indent=2))
        print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    _cli()
