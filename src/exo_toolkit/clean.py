"""Light curve cleaning: NaN removal, sigma-clipping, normalization, detrending.

Operates on a lightkurve.LightCurve object (received from fetch.py) and
returns a cleaned copy.  No lightkurve import is required at module load
time — only method calls on the passed-in object are made — so this
module works whether or not lightkurve is installed.

Pipeline (per step, each optional except NaN removal):
    1. remove_nans()              — always applied
    2. remove_outliers(sigma=…)   — disabled by sigma_clip=None
    3. normalize()                — disabled by normalize=False
    4. flatten(window_length=…)   — disabled by flatten=False

Public API
----------
clean_lightcurve(lc, *, sigma_clip, window_length, normalize, flatten)
    → CleanResult
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CleanProvenance(BaseModel):
    """Record of what cleaning steps were applied and how many points removed."""

    model_config = ConfigDict(frozen=True)

    n_cadences_raw: int = Field(ge=0)
    n_cadences_cleaned: int = Field(ge=0)
    n_removed_nan: int = Field(ge=0)
    n_removed_outlier: int = Field(ge=0)
    sigma_clip_sigma: float | None  # None when sigma-clipping is disabled
    window_length: int | None       # None when flattening is disabled
    normalized: bool
    flattened: bool


@dataclass(frozen=True)
class CleanResult:
    """A cleaned light curve and the provenance of what was done to it."""

    light_curve: Any  # lightkurve.LightCurve at runtime
    provenance: CleanProvenance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_lightcurve(
    lc: Any,
    *,
    sigma_clip: float | None = 5.0,
    window_length: int = 401,
    normalize: bool = True,
    flatten: bool = True,
) -> CleanResult:
    """Clean a light curve for BLS transit search.

    Steps are applied in order: NaN removal → sigma-clip → normalize →
    flatten.  Each step after NaN removal can be disabled independently.

    Args:
        lc: A lightkurve.LightCurve object (from fetch_lightcurve()).
        sigma_clip: Sigma threshold for outlier removal.  Pass None to
            skip the sigma-clipping step entirely.
        window_length: Savitzky-Golay window length for flatten().  Must
            be a positive odd integer.  Ignored when flatten=False.
        normalize: Divide flux by its median before flattening.  When
            flatten=True this is applied first; flatten() then removes
            the remaining long-term trend.
        flatten: Apply Savitzky-Golay detrending via lc.flatten().  This
            removes stellar variability and long-term systematics and
            implicitly normalizes the flux to ~1.0.

    Returns:
        CleanResult with the cleaned LightCurve and full provenance.

    Raises:
        ValueError: window_length is even (Savitzky-Golay requires odd).
    """
    if flatten and window_length % 2 == 0:
        raise ValueError(
            f"window_length must be a positive odd integer, got {window_length}"
        )

    n_raw = len(lc.time)

    # ------------------------------------------------------------------
    # Step 1: Remove NaN flux values (always applied)
    # ------------------------------------------------------------------
    lc_clean = lc.remove_nans()
    n_removed_nan = n_raw - len(lc_clean.time)

    # ------------------------------------------------------------------
    # Step 2: Sigma-clip outliers
    # ------------------------------------------------------------------
    n_removed_outlier = 0
    if sigma_clip is not None:
        lc_clean, outlier_mask = lc_clean.remove_outliers(
            sigma=sigma_clip, return_mask=True
        )
        n_removed_outlier = int(outlier_mask.sum())

    # ------------------------------------------------------------------
    # Step 3: Normalize flux to median 1.0
    # ------------------------------------------------------------------
    if normalize:
        lc_clean = lc_clean.normalize()

    # ------------------------------------------------------------------
    # Step 4: Savitzky-Golay detrend
    # ------------------------------------------------------------------
    if flatten:
        lc_clean = lc_clean.flatten(window_length=window_length)

    provenance = CleanProvenance(
        n_cadences_raw=n_raw,
        n_cadences_cleaned=len(lc_clean.time),
        n_removed_nan=n_removed_nan,
        n_removed_outlier=n_removed_outlier,
        sigma_clip_sigma=sigma_clip,
        window_length=window_length if flatten else None,
        normalized=normalize,
        flattened=flatten,
    )

    return CleanResult(light_curve=lc_clean, provenance=provenance)
