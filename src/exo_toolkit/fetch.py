"""Light curve fetching from MAST via Lightkurve.

Wraps Lightkurve's search → per-product download → stitch pattern into a
single call that returns a FetchResult containing a stitched LightCurve and
full provenance metadata.

Lightkurve is imported lazily inside fetch_lightcurve() so this module
loads without error even when lightkurve is not installed.  A clear
ImportError is raised at call time if the package is missing.

Public API
----------
fetch_lightcurve(target_id, mission, *, exptime, pipeline, sectors,
                 prefer_pdcsap)
    → FetchResult
"""
from __future__ import annotations

import datetime
import importlib
import importlib.util
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from exo_toolkit.schemas import Mission

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FetchProvenance(BaseModel):
    """Metadata captured at download time for reproducibility."""

    model_config = ConfigDict(frozen=True)

    target_id: str
    mission: Mission
    sectors_or_quarters: tuple[int, ...]
    cadence_seconds: float = Field(gt=0.0)
    pipeline: str
    flux_column: str
    n_cadences: int = Field(ge=1)
    time_baseline_days: float = Field(ge=0.0)
    fetched_at: str  # ISO 8601 UTC


@dataclass(frozen=True)
class FetchResult:
    """A stitched light curve and its download provenance."""

    light_curve: Any  # lightkurve.LightCurve at runtime
    provenance: FetchProvenance


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_DEFAULT_AUTHOR: dict[str, str] = {
    "TESS": "SPOC",
    "Kepler": "Kepler",
    "K2": "K2",
}

# lightkurve.search_lightcurve uses different parameter names per mission
_SECTOR_KWARG: dict[str, str] = {
    "TESS": "sector",
    "Kepler": "quarter",
    "K2": "campaign",
}

# Key in lc.meta that holds the sector / quarter / campaign number
_SECTOR_META_KEY: dict[str, str] = {
    "TESS": "SECTOR",
    "Kepler": "QUARTER",
    "K2": "CAMPAIGN",
}

# Path to the Skills directory (resolved once at import time)
_SKILLS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "Skills"

# Fallback cadence in seconds when lc.meta["EXPTIME"] is absent
_EXPTIME_FALLBACK: dict[str, float] = {
    "long": 1800.0,
    "short": 120.0,
    "fast": 20.0,
}

_DOWNLOAD_PRODUCTS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_lightcurve(
    target_id: str,
    mission: Mission,
    *,
    exptime: str = "long",
    pipeline: str | None = None,
    sectors: tuple[int, ...] | None = None,
    prefer_pdcsap: bool = True,
) -> FetchResult:
    """Download and stitch a light curve from MAST.

    Args:
        target_id: Target identifier — "TIC 123456789" for TESS,
            "KIC 123456789" for Kepler, "EPIC 123456789" for K2,
            or a JWST obsid (e.g. "jw01743001001_04101_00001_nrca1") for JWST.
        mission: "TESS", "Kepler", "K2", or "JWST".
        exptime: Exposure-time hint passed to lightkurve: "long" (20-min
            TESS / 30-min Kepler), "short" (2-min TESS), or "fast"
            (20-sec TESS).  Ignored for JWST.
        pipeline: Override the default pipeline author — e.g. "QLP"
            instead of the default "SPOC" for TESS.  Ignored for JWST.
        sectors: Restrict download to specific TESS sectors, Kepler
            quarters, or K2 campaigns.  None means all available.
            Ignored for JWST.
        prefer_pdcsap: Request PDCSAP (systematics-corrected) flux.
            Set False to use SAP flux instead.  Ignored for JWST.

    Returns:
        FetchResult containing the stitched LightCurve and provenance.

    Raises:
        ImportError: lightkurve is not installed (non-JWST missions).
        ValueError: No light curves found for the target / parameters.
    """
    if mission == "JWST":
        return _fetch_jwst(target_id)

    try:
        import lightkurve as lk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lightkurve is required for fetch operations. "
            "Install it with: pip install lightkurve"
        ) from exc

    author = pipeline or _DEFAULT_AUTHOR[mission]
    flux_columns = _flux_column_candidates(author, prefer_pdcsap=prefer_pdcsap)
    sector_kwarg = _SECTOR_KWARG[mission]
    sector_value: list[int] | None = list(sectors) if sectors is not None else None

    search = lk.search_lightcurve(
        target_id,
        mission=mission,
        exptime=exptime,
        author=author,
        **{sector_kwarg: sector_value},
    )

    if len(search) == 0:
        raise ValueError(
            f"No {mission} light curves found for {target_id!r} "
            f"(author={author!r}, exptime={exptime!r})"
        )

    collection, flux_columns_used = _download_collection_with_cache_repair(
        search,
        flux_columns=flux_columns,
    )
    # Lightkurve's default stitch() normalizes each product before stacking.
    # Keep project preprocessing order explicit: fetch raw selected flux,
    # then let clean_lightcurve() remove NaNs/outliers before normalizing.
    lc = collection.stitch(corrector_func=None)

    cadence_raw = lc.meta.get("EXPTIME", 0.0)
    cadence_seconds = (
        float(cadence_raw)
        if cadence_raw
        else _EXPTIME_FALLBACK.get(exptime, 1800.0)
    )

    actual_pipeline = str(lc.meta.get("PROCVER") or author)

    time_baseline_days = (
        float((lc.time[-1] - lc.time[0]).jd) if len(lc.time) >= 2 else 0.0
    )

    provenance = FetchProvenance(
        target_id=target_id,
        mission=mission,
        sectors_or_quarters=_extract_sectors(collection, mission),
        cadence_seconds=cadence_seconds,
        pipeline=actual_pipeline,
        flux_column="+".join(flux_columns_used),
        n_cadences=len(lc.time),
        time_baseline_days=time_baseline_days,
        fetched_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    return FetchResult(light_curve=lc, provenance=provenance)


# ---------------------------------------------------------------------------
# Provenance score
# ---------------------------------------------------------------------------

# Cadence bounds for linear interpolation of the cadence sub-score.
_CADENCE_BEST_S: float = 120.0    # 2-min TESS short-cadence
_CADENCE_WORST_S: float = 1800.0  # 30-min Kepler/TESS long-cadence

# Per-pipeline quality weights.  Unknown pipelines receive the default.
_PIPELINE_QUALITY: dict[str, float] = {
    "SPOC": 1.00,
    "QLP": 0.85,
    "TGLC": 0.75,
    "Kepler": 1.00,
    "K2": 1.00,
}
_PIPELINE_QUALITY_DEFAULT: float = 0.60

# Number of sectors / quarters needed for maximum sector sub-score.
_SECTORS_FOR_MAX: float = 3.0


def compute_provenance_score(provenance: FetchProvenance) -> float:
    """Compute a data-quality score in [0, 1] from download provenance metadata.

    The score is used as the ``provenance_score`` gate in
    :func:`~exo_toolkit.pathway.classify_submission_pathway`.  A score ≥ 0.80
    is required for the ``tfop_ready`` pathway.

    Sub-scores and weights
    ----------------------
    - **cadence** (0.40): linear ramp; 1.0 at 2-min, 0.0 at 30-min.
    - **sector coverage** (0.35): ``min(n_sectors / 3, 1.0)``; saturates at 3+ sectors.
    - **pipeline quality** (0.25): SPOC/Kepler/K2 → 1.0; QLP → 0.85; TGLC → 0.75;
      unknown → 0.60.
    """
    cadence_sub = max(
        0.0,
        min(
            1.0,
            (_CADENCE_WORST_S - provenance.cadence_seconds)
            / (_CADENCE_WORST_S - _CADENCE_BEST_S),
        ),
    )
    sector_sub = min(len(provenance.sectors_or_quarters) / _SECTORS_FOR_MAX, 1.0)
    pipeline_sub = _PIPELINE_QUALITY.get(provenance.pipeline, _PIPELINE_QUALITY_DEFAULT)
    return 0.40 * cadence_sub + 0.35 * sector_sub + 0.25 * pipeline_sub


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_jwst(obsid: str, *, _skills_dir: Path | None = None) -> FetchResult:
    """Fetch a JWST light curve via Skills/fetch_jwst_lc.py.

    Args:
        obsid: JWST observation ID (e.g. "jw01743001001_04101_00001_nrca1").
        _skills_dir: Override Skills directory path (for tests only).

    Returns:
        FetchResult with a lightkurve-compatible LightCurve object.

    Raises:
        ImportError: fetch_jwst_lc.py cannot be loaded from the Skills directory.
        ValueError: No JWST data found for the given obsid.
    """
    skills_dir = _skills_dir or _SKILLS_DIR
    jwst_path = skills_dir / "fetch_jwst_lc.py"
    if not jwst_path.exists():
        raise ImportError(f"Cannot find fetch_jwst_lc.py at {jwst_path}")
    spec = importlib.util.spec_from_file_location("fetch_jwst_lc", jwst_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fetch_jwst_lc.py from {jwst_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.fetch_jwst_lc(obsid)
    if result is None:
        raise ValueError(f"No JWST data found for obsid {obsid!r}")

    lc = module.to_lightkurve(result)

    time_arr: list[float] = result.time_btjd
    cadence_seconds = (
        float((time_arr[-1] - time_arr[0]) / (len(time_arr) - 1)) * 86400.0
        if len(time_arr) >= 2
        else 60.0
    )
    baseline_days = float(time_arr[-1] - time_arr[0]) if len(time_arr) >= 2 else 0.0

    provenance = FetchProvenance(
        target_id=obsid,
        mission="JWST",
        sectors_or_quarters=(),
        cadence_seconds=max(cadence_seconds, 1.0),
        pipeline="JWST",
        flux_column=result.product_type,
        n_cadences=result.n_integrations,
        time_baseline_days=baseline_days,
        fetched_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    return FetchResult(light_curve=lc, provenance=provenance)


def _flux_column_candidates(author: str, *, prefer_pdcsap: bool) -> tuple[str, ...]:
    """Return flux-column candidates for a Lightkurve author/pipeline."""
    if author.upper() == "QLP":
        if not prefer_pdcsap:
            return ("sap_flux",)
        # QLP does not provide PDCSAP_FLUX.  Older products use KSPSAP_* for
        # corrected flux; newer products use DET_* and sometimes SYS_RM_FLUX.
        return ("kspsap_flux", "det_flux", "sys_rm_flux", "sap_flux")
    return ("pdcsap_flux",) if prefer_pdcsap else ("sap_flux",)


def _download_collection_with_cache_repair(
    search: Any,
    *,
    flux_columns: tuple[str, ...],
) -> tuple[Any, tuple[str, ...]]:
    """Download a Lightkurve search result without mutating global stdout.

    ``SearchResult.download()`` and ``download_all()`` are decorated with
    Lightkurve's ``suppress_stdout`` helper, which assigns ``sys.stdout``
    process-wide. That is unsafe while the discovery scanner runs multiple
    worker threads and the main thread prints progress. Use the lower-level
    per-product downloader instead, and still repair corrupt cached FITS files
    when Lightkurve reports an interrupted download.
    """
    try:
        import lightkurve as lk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lightkurve is required for fetch operations. "
            "Install it with: pip install lightkurve"
        ) from exc

    light_curves: list[Any] = []
    flux_columns_used: list[str] = []
    for idx in range(len(search.table)):
        light_curve, flux_column = _download_one_with_cache_repair(
            search,
            table=search.table[idx : idx + 1],
            flux_columns=flux_columns,
        )
        light_curves.append(light_curve)
        if flux_column not in flux_columns_used:
            flux_columns_used.append(flux_column)
    if not light_curves:
        raise ValueError("No downloadable light curves returned by Lightkurve")
    return lk.LightCurveCollection(light_curves), tuple(flux_columns_used)


def _download_one_with_cache_repair(
    search: Any,
    *,
    table: Any,
    flux_columns: tuple[str, ...],
) -> tuple[Any, str]:
    """Download one Lightkurve product, repairing one corrupt cache file at a time."""
    max_cache_repairs = 4
    last_missing_flux_error: Exception | None = None
    for flux_column in flux_columns:
        for attempt in range(max_cache_repairs + 1):
            try:
                return (
                    _download_one_quietly(
                        search,
                        table=table,
                        quality_bitmask="default",
                        download_dir=None,
                        cutout_size=None,
                        flux_column=flux_column,
                    ),
                    flux_column,
                )
            except Exception as exc:
                if _is_missing_flux_column_error(exc, flux_column):
                    last_missing_flux_error = exc
                    break
                repaired = _remove_corrupt_lightkurve_cache_file(exc)
                if repaired is not None and attempt < max_cache_repairs:
                    continue
                raise

    if last_missing_flux_error is not None:
        raise last_missing_flux_error
    raise RuntimeError("Lightkurve cache repair retry loop exhausted")


def _download_one_quietly(search: Any, **kwargs: Any) -> Any:
    """Call Lightkurve's per-product downloader with Astroquery verbosity disabled.

    Lightkurve's lower-level ``_download_one`` avoids its ``suppress_stdout``
    decorator, but it still calls ``Observations.download_products`` with
    Astroquery's default ``verbose=True``.  Force ``verbose=False`` while the
    third-party call is active so worker-thread scans keep operator progress
    readable without redirecting process-global stdout.
    """
    mast_module: Any = importlib.import_module("astroquery.mast")
    observations: Any = mast_module.Observations
    original = observations.download_products

    def quiet_download_products(products: Any, *args: Any, **inner_kwargs: Any) -> Any:
        inner_kwargs["verbose"] = False
        return original(products, *args, **inner_kwargs)

    with _DOWNLOAD_PRODUCTS_LOCK:
        observations.download_products = quiet_download_products
        try:
            return search._download_one(**kwargs)  # noqa: SLF001
        finally:
            observations.download_products = original


def _is_missing_flux_column_error(exc: Exception, flux_column: str) -> bool:
    """Return True when Lightkurve wrapped a missing FITS flux column."""
    cause = exc.__cause__
    return isinstance(cause, KeyError) and str(cause).strip("'\"") == flux_column


def _remove_corrupt_lightkurve_cache_file(exc: Exception) -> Path | None:
    """Remove one corrupt Lightkurve MAST cache FITS named by *exc*."""
    text = str(exc)
    if "This file may be corrupt due to an interrupted download" not in text:
        return None
    for raw_path in _extract_fits_paths(text):
        candidate = Path(raw_path)
        if (
            candidate.suffix == ".fits"
            and ".lightkurve" in candidate.parts
            and "mastDownload" in candidate.parts
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                return candidate
            if not candidate.exists():
                return candidate
    return None


def _extract_fits_paths(text: str) -> list[str]:
    """Return absolute FITS paths embedded in a Lightkurve error string."""
    paths = re.findall(r"(/[^\n]+?\.fits)", text)
    return [path.strip() for path in paths]


def _extract_sectors(collection: Any, mission: str) -> tuple[int, ...]:
    """Collect unique, sorted sector/quarter/campaign numbers from a collection."""
    key = _SECTOR_META_KEY[mission]
    seen: list[int] = []
    for lc in collection:
        val = lc.meta.get(key)
        if val is not None:
            n = int(val)
            if n not in seen:
                seen.append(n)
    return tuple(sorted(seen))
