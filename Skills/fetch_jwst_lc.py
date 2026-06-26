"""Download JWST time-series data from MAST and convert to pipeline LightCurve format.

This is Option A2 of the DISCOVERY_RUNBOOK: given a JWST observation (obsid from
fetch_jwst_targets.py), download the best available data product, extract a
white-light flux time series, and return it as a lightkurve LightCurve object
compatible with the existing clean → search → vet pipeline.

Product priority order (best to worst):
  1. ``_x1dints.fits``   — Stage 3 NIRISS SOSS: 1D extracted spectra per integration;
                           white-light curve formed by summing flux across wavelength
  2. ``_calints.fits``   — Stage 2 calibrated integrations; white-light from rate images
                           (less clean; use when Stage 3 unavailable)

Time conversion: JWST uses MJD (Modified Julian Date). Pipeline uses BTJD (BJD−2457000).
Conversion: BTJD = MJD − 51544.5 + 2400000.5 − 2457000 = MJD − 109543.0

Usage::

    # Fetch white-light curve for a specific JWST obsid
    .venv/bin/python Skills/fetch_jwst_lc.py 87654321 --output data/jwst_wl.json

    # Run through pipeline immediately
    .venv/bin/python Skills/fetch_jwst_lc.py 87654321 --run-pipeline

    # Batch mode: process all obsids from fetch_jwst_targets.py output
    .venv/bin/python Skills/fetch_jwst_lc.py \\
        --from-targets data/jwst_timeseries_targets.json \\
        --output-dir data/jwst_lcs/

Public API
----------
JwstLcResult      dataclass: obsid, target_name, time_btjd, flux_norm, flux_err_norm,
                  instrument, n_integrations, product_type, warnings
fetch_jwst_lc(obsid, *, product_fn, download_fn) -> JwstLcResult | None
white_light_from_x1dints(fits_path) -> tuple[ndarray, ndarray, ndarray]
white_light_from_calints(fits_path) -> tuple[ndarray, ndarray, ndarray]
to_lightkurve(result) -> LightCurve  (requires lightkurve installed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BTJD = BJD - 2457000; BJD ≈ MJD + 2400000.5; so BTJD = MJD - 56999.5
_MJD_TO_BTJD_OFFSET = 56999.5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JwstLcResult:
    obsid: str
    target_name: str
    time_btjd: list[float]       # BTJD timestamps
    flux_norm: list[float]       # normalized to median = 1.0
    flux_err_norm: list[float]   # normalized errors
    instrument: str
    n_integrations: int
    product_type: str            # "x1dints" | "calints"
    warnings: list[str]


# ---------------------------------------------------------------------------
# FITS extraction helpers
# ---------------------------------------------------------------------------

def white_light_from_x1dints(fits_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract white-light time series from a JWST _x1dints.fits file.

    The x1dints format stores one FITS extension per integration, each containing
    a table with columns WAVELENGTH, FLUX, ERROR. We sum flux across all wavelength
    bins (within a configurable range) to form the white-light curve.

    Returns:
        (time_mjd, flux_wl, flux_err_wl) — one entry per integration.
    """
    from astropy.io import fits  # type: ignore[import]

    times: list[float] = []
    fluxes: list[float] = []
    errors: list[float] = []

    with fits.open(str(fits_path)) as hdul:
        # Extension 0: primary header (may have TIME-MID keyword per integration)
        # Extensions 1+: one per integration (BinTableHDU with WAVELENGTH/FLUX/ERROR)
        for ext in hdul[1:]:
            if ext.name in ("PRIMARY", "ASDF"):
                continue
            # Get mid-time from extension header (MJD)
            t_mid = ext.header.get("MJD-MID") or ext.header.get("MJD-BEG") or ext.header.get("MJD-AVG")
            if t_mid is None:
                continue
            try:
                data = ext.data
                wave = data["WAVELENGTH"]
                flux = data["FLUX"]
                err = data.get("ERROR") or data.get("FLUX_ERROR") or np.ones_like(flux)
            except (KeyError, AttributeError):
                continue
            # Mask non-finite
            mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)
            if mask.sum() < 10:
                continue
            times.append(float(t_mid))
            fluxes.append(float(np.nansum(flux[mask])))
            # Error on sum: sqrt(sum of variances)
            errors.append(float(np.sqrt(np.nansum((err[mask]) ** 2))))

    t = np.array(times)
    f = np.array(fluxes)
    e = np.array(errors)
    # Sort by time
    order = np.argsort(t)
    return t[order], f[order], e[order]


def white_light_from_calints(fits_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract white-light time series from a JWST _calints.fits file.

    The calints format stores calibrated 2D (or 3D) integrations in a single
    image extension. We perform aperture photometry by summing flux in each
    integration frame.

    Returns:
        (time_mjd, flux_wl, flux_err_wl) — one entry per integration.
    """
    from astropy.io import fits  # type: ignore[import]

    with fits.open(str(fits_path)) as hdul:
        # SCI extension contains the data cube: (n_integrations, n_rows, n_cols)
        # ERR extension contains the error cube
        sci = hdul["SCI"].data
        try:
            err = hdul["ERR"].data
        except KeyError:
            err = np.sqrt(np.abs(sci))

        # TIME extension or INT_TIMES table for timestamps
        times_mjd: np.ndarray | None = None
        for ext_name in ("INT_TIMES", "TIME", "TIMES"):
            try:
                int_times = hdul[ext_name]
                # Look for MJD column
                for col in ("int_mid_MJD_UTC", "int_mid_BJD_TDB", "MJD-MID", "MJD"):
                    if col in int_times.columns.names:
                        times_mjd = np.array(int_times.data[col], dtype=float)
                        break
                if times_mjd is not None:
                    break
            except (KeyError, AttributeError):
                continue

        if times_mjd is None:
            # Reconstruct from header: TSTART, NGROUPS, TGROUP, TFRAME
            hdr = hdul["SCI"].header
            tstart = float(hdr.get("TSTART", 0.0))
            n_int = sci.shape[0] if sci.ndim >= 3 else 1
            # Approximate: evenly spaced
            tend = float(hdr.get("TEND", tstart + 0.1))
            times_mjd = np.linspace(tstart, tend, n_int)

    # Sum spatial pixels in each integration for white-light flux
    n_int = sci.shape[0] if sci.ndim == 3 else 1
    if sci.ndim == 2:
        sci = sci[np.newaxis]
        err = err[np.newaxis]

    fluxes = np.zeros(n_int)
    flux_errs = np.zeros(n_int)
    for i in range(n_int):
        frame = sci[i]
        e_frame = err[i]
        mask = np.isfinite(frame) & np.isfinite(e_frame) & (e_frame > 0)
        fluxes[i] = float(np.nansum(frame[mask]))
        flux_errs[i] = float(np.sqrt(np.nansum(e_frame[mask] ** 2)))

    order = np.argsort(times_mjd)
    return times_mjd[order], fluxes[order], flux_errs[order]


def _normalize(flux: np.ndarray, flux_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize to median = 1.0."""
    med = float(np.nanmedian(flux))
    if med == 0 or not np.isfinite(med):
        return flux, flux_err
    return flux / med, flux_err / med


# ---------------------------------------------------------------------------
# MAST download helpers
# ---------------------------------------------------------------------------

def _default_product_fn(obsid: str) -> list[dict[str, str]]:
    """Return list of {filename, dataURI, type} for an obsid."""
    from astroquery.mast import Observations  # type: ignore[import]

    try:
        products = Observations.get_product_list([obsid])
    except Exception:  # noqa: BLE001
        return []
    result = []
    for row in products:
        result.append({
            "filename": str(row.get("productFilename", "")),
            "dataURI": str(row.get("dataURI", "")),
            "description": str(row.get("description", "")),
            "type": str(row.get("productType", "")),
        })
    return result


def _default_download_fn(data_uri: str, dest: Path) -> Path:
    """Download a MAST data product to dest using astroquery."""
    from astroquery.mast import Observations  # type: ignore[import]

    manifest = Observations.download_products(
        [{"dataURI": data_uri}],
        download_dir=str(dest.parent),
    )
    # astroquery saves to ./mastDownload/... subdirectory; find it
    if manifest is not None and len(manifest) > 0:
        local_path = Path(str(manifest["Local Path"][0]))
        if local_path.exists():
            return local_path
    raise FileNotFoundError(f"Download of {data_uri} failed")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_jwst_lc(
    obsid: str,
    *,
    target_name: str = "",
    instrument: str = "",
    cache_dir: Path | None = None,
    product_fn: Callable[[str], list[dict[str, str]]] | None = None,
    download_fn: Callable[[str, Path], Path] | None = None,
) -> JwstLcResult | None:
    """Download and normalize JWST time-series data for one observation.

    Args:
        obsid: MAST observation ID (from fetch_jwst_targets.py output).
        target_name: Human-readable target name (for output labeling).
        instrument: Instrument name (used for logging only).
        cache_dir: Directory to cache downloaded FITS files. Default: data/jwst_cache/.
        product_fn: Injectable product-list function for testing.
        download_fn: Injectable download function for testing.

    Returns:
        JwstLcResult if successful, None if no suitable products are found.
    """
    cache_dir = cache_dir or Path("data/jwst_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    pfn = product_fn or _default_product_fn
    dfn = download_fn or _default_download_fn

    products = pfn(obsid)
    if not products:
        return None

    # Sort by preference: x1dints > calints
    def _priority(p: dict[str, str]) -> int:
        fn = p["filename"].lower()
        if "x1dints" in fn:
            return 0
        if "calints" in fn:
            return 1
        return 99

    candidates = sorted(
        [p for p in products if any(k in p["filename"].lower() for k in ("x1dints", "calints"))],
        key=_priority,
    )
    if not candidates:
        warnings.append(f"No x1dints or calints products found for obsid {obsid}")
        return None

    best = candidates[0]
    product_type = "x1dints" if "x1dints" in best["filename"].lower() else "calints"
    local_path = cache_dir / best["filename"]

    # Download if not already cached
    if not local_path.exists():
        try:
            local_path = dfn(best["dataURI"], local_path)
        except Exception as exc:
            warnings.append(f"Download failed: {exc}")
            return None

    # Extract white-light curve
    try:
        if product_type == "x1dints":
            time_mjd, flux_raw, err_raw = white_light_from_x1dints(local_path)
        else:
            time_mjd, flux_raw, err_raw = white_light_from_calints(local_path)
    except Exception as exc:
        warnings.append(f"Extraction failed: {exc}")
        return None

    if len(time_mjd) < 10:
        warnings.append(f"Too few integrations ({len(time_mjd)}) — skipping")
        return None

    # Convert MJD → BTJD
    time_btjd = time_mjd - _MJD_TO_BTJD_OFFSET

    # Remove non-finite
    mask = np.isfinite(flux_raw) & np.isfinite(err_raw) & (err_raw > 0)
    if mask.sum() < 10:
        warnings.append("Too few finite integrations after masking")
        return None

    time_btjd = time_btjd[mask]
    flux_raw = flux_raw[mask]
    err_raw = err_raw[mask]

    flux_norm, err_norm = _normalize(flux_raw, err_raw)

    return JwstLcResult(
        obsid=obsid,
        target_name=target_name or obsid,
        time_btjd=time_btjd.tolist(),
        flux_norm=flux_norm.tolist(),
        flux_err_norm=err_norm.tolist(),
        instrument=instrument,
        n_integrations=int(mask.sum()),
        product_type=product_type,
        warnings=warnings,
    )


def to_lightkurve(result: JwstLcResult) -> Any:
    """Convert JwstLcResult to a lightkurve LightCurve object.

    The returned object is compatible with clean.py, search.py, and vet.py.
    Requires lightkurve to be installed.
    """
    import lightkurve as lk  # type: ignore[import]
    from astropy.time import Time  # type: ignore[import]
    import astropy.units as u  # type: ignore[import]

    t = Time(np.array(result.time_btjd) + 2457000, format="jd", scale="tdb")
    lc = lk.LightCurve(
        time=t,
        flux=np.array(result.flux_norm) * u.dimensionless_unscaled,
        flux_err=np.array(result.flux_err_norm) * u.dimensionless_unscaled,
        meta={
            "LABEL": result.target_name,
            "MISSION": "JWST",
            "INSTRUMENT": result.instrument,
            "OBSID": result.obsid,
        },
    )
    return lc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Download JWST time-series data and convert to pipeline LightCurve format"
    )
    parser.add_argument("obsid", nargs="?", help="MAST observation ID")
    parser.add_argument("--from-targets", help="JSON file from fetch_jwst_targets.py")
    parser.add_argument("--output", help="Write JwstLcResult JSON to this path")
    parser.add_argument("--output-dir", help="Directory for batch output (one JSON per obsid)")
    parser.add_argument("--cache-dir", default="data/jwst_cache", help="FITS cache directory")
    parser.add_argument("--run-pipeline", action="store_true",
                        help="Run clean+search+vet on the downloaded light curve")
    args = parser.parse_args()

    obsids: list[tuple[str, str, str]] = []  # (obsid, target_name, instrument)

    if args.from_targets:
        targets = json.loads(Path(args.from_targets).read_text())
        obsids = [(t["obsid"], t.get("target_name", ""), t.get("instrument", "")) for t in targets]
    elif args.obsid:
        obsids = [(args.obsid, "", "")]
    else:
        parser.print_help()
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    n = len(obsids)
    ok = 0
    start = time.monotonic()

    for i, (obsid, tname, inst) in enumerate(obsids, 1):
        elapsed = time.monotonic() - start
        rate = i / elapsed if elapsed > 0 else float("inf")
        remaining = (n - i) / rate if rate > 0 else float("inf")
        eta = f"{remaining/60:.0f}m{remaining%60:.0f}s" if remaining > 90 else f"{remaining:.0f}s"
        print(f"  [{i}/{n}]  {obsid}  elapsed={elapsed:.0f}s  ETA={eta}", flush=True)

        result = fetch_jwst_lc(obsid, target_name=tname, instrument=inst, cache_dir=cache_dir)
        if result is None:
            print(f"    SKIP: no suitable products", flush=True)
            continue

        ok += 1
        if result.warnings:
            for w in result.warnings:
                print(f"    WARN: {w}", flush=True)
        print(
            f"    OK: {result.n_integrations} integrations via {result.product_type}",
            flush=True,
        )

        # Write output
        if args.output and len(obsids) == 1:
            Path(args.output).write_text(json.dumps(asdict(result), indent=2))
        elif args.output_dir:
            out = Path(args.output_dir) / f"{obsid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(asdict(result), indent=2))

        if args.run_pipeline:
            _run_pipeline(result)

    print(f"\nDone: {ok}/{n} observations downloaded successfully", flush=True)


def _run_pipeline(result: JwstLcResult) -> None:
    """Run clean → search → vet on a downloaded JWST light curve."""
    import sys
    sys.path.insert(0, "src")
    from exo_toolkit.clean import clean_lightcurve
    from exo_toolkit.search import search_lightcurve

    try:
        lc = to_lightkurve(result)
    except ImportError:
        print("    PIPELINE: lightkurve not installed — skipping", flush=True)
        return

    print(f"    PIPELINE: cleaning {result.n_integrations} points...", flush=True)
    clean_result = clean_lightcurve(lc)
    lc_clean = clean_result.light_curve

    print(f"    PIPELINE: searching for transits...", flush=True)
    signals = search_lightcurve(lc_clean, min_snr=5.0, max_peaks=5)
    if not signals:
        print(f"    PIPELINE: no signals found above SNR 5.0", flush=True)
    else:
        for s in signals:
            print(
                f"    PIPELINE: signal P={s.period_days:.3f}d  SNR={s.snr:.1f}  "
                f"depth={s.depth_ppm:.0f}ppm",
                flush=True,
            )


if __name__ == "__main__":
    _cli()
