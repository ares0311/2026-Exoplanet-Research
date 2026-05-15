"""Light curve fetching from MAST via Lightkurve.

Wraps Lightkurve's search → download_all → stitch pattern into a single
call that returns a FetchResult containing a stitched LightCurve and full
provenance metadata.

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
from dataclasses import dataclass
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

_DEFAULT_AUTHOR: dict[Mission, str] = {
    "TESS": "SPOC",
    "Kepler": "Kepler",
    "K2": "K2",
}

# lightkurve.search_lightcurve uses different parameter names per mission
_SECTOR_KWARG: dict[Mission, str] = {
    "TESS": "sector",
    "Kepler": "quarter",
    "K2": "campaign",
}

# Key in lc.meta that holds the sector / quarter / campaign number
_SECTOR_META_KEY: dict[Mission, str] = {
    "TESS": "SECTOR",
    "Kepler": "QUARTER",
    "K2": "CAMPAIGN",
}

# Fallback cadence in seconds when lc.meta["EXPTIME"] is absent
_EXPTIME_FALLBACK: dict[str, float] = {
    "long": 1800.0,
    "short": 120.0,
    "fast": 20.0,
}


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
            "KIC 123456789" for Kepler, "EPIC 123456789" for K2.
        mission: "TESS", "Kepler", or "K2".
        exptime: Exposure-time hint passed to lightkurve: "long" (20-min
            TESS / 30-min Kepler), "short" (2-min TESS), or "fast"
            (20-sec TESS).
        pipeline: Override the default pipeline author — e.g. "QLP"
            instead of the default "SPOC" for TESS.
        sectors: Restrict download to specific TESS sectors, Kepler
            quarters, or K2 campaigns.  None means all available.
        prefer_pdcsap: Request PDCSAP (systematics-corrected) flux.
            Set False to use SAP flux instead.

    Returns:
        FetchResult containing the stitched LightCurve and provenance.

    Raises:
        ImportError: lightkurve is not installed.
        ValueError: No light curves found for the target / parameters.
    """
    try:
        import lightkurve as lk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lightkurve is required for fetch operations. "
            "Install it with: pip install lightkurve"
        ) from exc

    author = pipeline or _DEFAULT_AUTHOR[mission]
    flux_column = "pdcsap_flux" if prefer_pdcsap else "sap_flux"
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

    collection = search.download_all(flux_column=flux_column)
    lc = collection.stitch()

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
        flux_column=flux_column,
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


def _extract_sectors(collection: Any, mission: Mission) -> tuple[int, ...]:
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
