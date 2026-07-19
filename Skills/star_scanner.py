"""Optimal star selection and background scanning for exoplanet transit search.

Four operating modes:

  1. **Single-target** (``--target <TIC_ID>``): analyse one star, log result.
  2. **Background scan** (default): query the TESS Input Catalog for
     high-priority uncharacterised stars, rank them, and scan in order until
     stopped (Ctrl-C) or ``--max-stars`` is reached.
  3. **Prepare only** (``--prepare-only``): freeze metadata and manifests.
  4. **Prepared execution** (``--execute-prepared-batch``): scan only the
     immutable queue and write schema-v2 candidate-ledger outcomes.

Stars already in the TESS TOI disposition list (known objects being actively
followed up) and stars already present in the scan log are skipped
automatically.  Scanning can be interrupted and resumed at any time — the log
records which TIC IDs have been processed.

Usage
-----
    # Single target
    .venv/bin/python Skills/star_scanner.py --target 150428135

    # Background scan (auto-resumes via log)
    .venv/bin/python Skills/star_scanner.py --log logs/scan_log.json --max-stars 1000

    # Show scan log summary without scanning
    .venv/bin/python Skills/star_scanner.py --summary --log logs/scan_log.json

    # Narrow magnitude window; use ML scorer
    .venv/bin/python Skills/star_scanner.py --log logs/scan_log.json \\
        --tmag-min 11 --tmag-max 13 \\
        --scorer xgboost --model-path data/model.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from exo_toolkit import __version__  # noqa: E402
from exo_toolkit.candidate_ledger import CandidateLedgerRecord  # noqa: E402
from exo_toolkit.cli import run_pipeline  # noqa: E402
from exo_toolkit.dataset_manifest import (  # noqa: E402
    DatasetManifest,
    load_dataset_manifest,
    sha256_file,
    validate_dataset_manifest,
)
from Skills.candidate_database import CandidateDatabase  # noqa: E402
from Skills.run_report import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    RunReport,
    report_path_for,
    run_and_commit_report,
)

# ExoFOP TOI table (same endpoint used by fetch_tess_toi.py)
_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

_SKILLS_DIR = Path(__file__).resolve().parent

_DEFAULT_SEARCH_CENTERS: tuple[tuple[float, float], ...] = tuple(
    (float(ra), float(dec))
    for dec in (-60, -40, -20, 0, 20, 40, 60)
    for ra in range(0, 360, 20)
)
# 7 declination bands x 18 right-ascension steps = 126 tiles (0.5 deg radius
# each => ~99 sq deg total coverage, ~0.24% of the full 41,253 sq deg sky).
# Roughly double the density of the original 5x12=60-tile grid in both axes.
# This is still a documented sample, not an exhaustive survey -- see
# select_targets()'s search_log output for the exact coverage of any given
# run.


class _StartRateLimiter:
    """Space worker starts so live-service requests do not burst at once."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = max(0.0, delay_seconds)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        """Wait until the next start slot is available."""
        if self._delay_seconds <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._delay_seconds
        if wait_seconds > 0.0:
            time.sleep(wait_seconds)


def _url_text(url: str, *, timeout: int = 60) -> str:
    """Fetch URL text using certifi when available."""
    import ssl

    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None

    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _row_get(row: Any, key: str) -> Any:
    """Return a value from dict-like or astropy table rows."""
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, TypeError, ValueError):
        return None


def _format_eta(seconds: float) -> str:
    """Format an ETA duration for console progress."""
    if seconds == float("inf"):
        return "unknown"
    if seconds > 90:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m{remainder:02d}s"
    return f"{seconds:.0f}s"


def _status_text(result: dict[str, Any]) -> str:
    """Return a compact console status for a scan result."""
    status = result["status"]
    if status == "candidate_found":
        return (
            f"CANDIDATE P={result['best_period_days']:.2f} d "
            f"FPP={result['best_fpp']:.3f} [{result['best_pathway']}]"
        )
    if status == "scanned_clear":
        return "clear"
    if status == "no_data":
        return "no data"
    return f"error: {result['error_message']}"


def _progress_suffix(done: int, total: int, start_time: float) -> str:
    """Return elapsed/ETA suffix for per-target progress output."""
    elapsed = time.monotonic() - start_time
    rate = done / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - done) / rate if rate > 0.0 else float("inf")
    return f"elapsed={elapsed:.0f}s ETA={_format_eta(remaining)}"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically so preparation artifacts cannot be left partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def priority_score(
    tmag: float,
    teff: float | None = None,
    n_sectors: int | None = None,
    contratio: float | None = None,
    radius_rsun: float | None = None,
) -> float:
    """Compute a [0, 1] priority score for a TIC star.

    Higher score = more promising target for planet transit search.

    Weighted sub-scores:

    * **Magnitude** (0.25): peaks at Tmag ≈ 12–13; falls off outside [10, 14].
    * **Stellar type** (0.20): prefers K/M dwarfs (3000–5500 K) where the
      habitable zone lies at short, easily-observed periods.
    * **Sector coverage** (0.20): more sectors → more transits visible; capped
      at 6 sectors for score = 1.0.  ``None`` → neutral 0.5.
    * **Contamination** (0.15): lower contamination → cleaner transit depth;
      ``contratio = 0`` scores 1.0.  ``None`` → neutral 0.5.
    * **Stellar radius** (0.20): smaller stars → deeper transit for a
      fixed-size planet (transit depth scales as ``(R_planet / R_star)^2``),
      so a given planet is far easier to detect around a small star.
      Peaks for M/K dwarfs (``R <= 0.7`` R_sun); falls off for larger stars.
      ``None`` → neutral 0.5.

    Args:
        tmag: TESS magnitude.
        teff: Effective temperature in Kelvin (``None`` → neutral 0.5).
        n_sectors: Sectors of TESS data available (``None`` → neutral 0.5).
        contratio: Fraction of aperture flux from nearby sources
            (``None`` → neutral 0.5).
        radius_rsun: Stellar radius in solar radii (``None`` → neutral 0.5).

    Returns:
        Priority in [0, 1].
    """
    # Magnitude score — ramp up 8→10, flat 10→13, ramp down 13→16
    if tmag <= 8.0:
        mag_score = 0.0
    elif tmag <= 10.0:
        mag_score = (tmag - 8.0) / 2.0 * 0.5          # 0.0 → 0.5
    elif tmag <= 12.0:
        mag_score = 0.5 + (tmag - 10.0) / 2.0 * 0.5   # 0.5 → 1.0
    elif tmag <= 13.0:
        mag_score = 1.0
    elif tmag <= 14.0:
        mag_score = 1.0 - (tmag - 13.0) * 0.5          # 1.0 → 0.5
    elif tmag <= 16.0:
        mag_score = 0.5 - (tmag - 14.0) / 2.0 * 0.5   # 0.5 → 0.0
    else:
        mag_score = 0.0

    # Stellar-type score
    if teff is None:
        teff_score = 0.5
    elif teff < 3000.0:
        teff_score = 0.3
    elif teff <= 4500.0:
        teff_score = 1.0   # M dwarf
    elif teff <= 5500.0:
        teff_score = 0.9   # K dwarf
    elif teff <= 6000.0:
        teff_score = 0.6   # solar-type G
    elif teff <= 7000.0:
        teff_score = 0.3   # F star
    else:
        teff_score = 0.1   # hot star

    # Sector-coverage score
    sector_score = 0.5 if n_sectors is None else min(n_sectors / 6.0, 1.0)

    # Contamination score
    cont_score = 0.5 if contratio is None else max(0.0, 1.0 - min(float(contratio), 1.0))

    # Stellar-radius score — smaller stars give deeper, more detectable
    # transits for a fixed planet size.
    if radius_rsun is None:
        radius_score = 0.5
    elif radius_rsun <= 0.7:
        radius_score = 1.0   # M/K dwarf
    elif radius_rsun <= 1.0:
        radius_score = 0.7   # G dwarf, Sun-like
    elif radius_rsun <= 1.5:
        radius_score = 0.4   # F dwarf / subgiant
    else:
        radius_score = 0.15  # giant — transits nearly undetectable

    return (
        0.25 * mag_score
        + 0.20 * teff_score
        + 0.20 * sector_score
        + 0.15 * cont_score
        + 0.20 * radius_score
    )


# ---------------------------------------------------------------------------
# Persistent scan log
# ---------------------------------------------------------------------------


class ScanLog:
    """Read/write a JSON scan log tracking which stars have been analysed.

    Log schema::

        {
          "last_updated": "<ISO-8601 UTC>",
          "active": {
            "<tic_id>": {
              "tic_id": int,
              "started_at": "<ISO-8601 UTC>",
              "priority_score": float | null,
              "pipeline": str | null,
              "exptime": str | null
            }
          },
          "entries": {
            "<tic_id>": {
              "tic_id": int,
              "scanned_at": "<ISO-8601 UTC>",
              "status": "candidate_found|scanned_clear|no_data|error",
              "n_signals": int,
              "best_period_days": float | null,
              "best_fpp": float | null,
              "best_pathway": str | null,
              "best_snr": float | null,
              "best_detection_confidence": float | null,
              "best_novelty_score": float | null,
              "best_depth_ppm": float | null,
              "best_duration_hours": float | null,
              "best_transit_count": int | null,
              "provenance_score": float | null,
              "signals": list[dict],
              "priority_score": float | null,
              "pipeline": str | null,
              "exptime": str | null,
              "error_message": str | null
            }
          }
        }

    Writes are atomic: the file is written to a ``.tmp`` sibling and then
    renamed, so a partial write never corrupts the log.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"last_updated": "", "entries": {}}
        if path.exists():
            with path.open() as fh:
                self._data = json.load(fh)
        self._data.setdefault("entries", {})
        self._data["active"] = {}
        self._data["last_updated"] = datetime.now(UTC).isoformat()
        self._flush()

    def is_scanned(self, tic_id: int) -> bool:
        """Return True if *tic_id* has an entry in the log."""
        with self._lock:
            return str(tic_id) in self._data["entries"]

    def scanned_ids(self) -> set[int]:
        """Return the set of all TIC IDs recorded in the log."""
        with self._lock:
            return {int(k) for k in self._data["entries"]}

    def mark_started(
        self,
        tic_id: int,
        target: dict[str, Any],
        *,
        pipeline: str,
        exptime: str,
    ) -> None:
        """Record that a worker has started *tic_id* without making it a resume skip."""
        with self._lock:
            self._data.setdefault("active", {})[str(tic_id)] = {
                "tic_id": tic_id,
                "started_at": datetime.now(UTC).isoformat(),
                "priority_score": target.get("priority"),
                "pipeline": pipeline,
                "exptime": exptime,
            }
            self._data["last_updated"] = datetime.now(UTC).isoformat()
            self._flush()

    def record(self, tic_id: int, status: str, result: dict[str, Any]) -> None:
        """Add or overwrite the log entry for *tic_id*."""
        with self._lock:
            entry: dict[str, Any] = {
                "tic_id": tic_id,
                "scanned_at": datetime.now(UTC).isoformat(),
                "status": status,
                "n_signals": result.get("n_signals", 0),
                "best_period_days": result.get("best_period_days"),
                "best_fpp": result.get("best_fpp"),
                "best_pathway": result.get("best_pathway"),
                "best_snr": result.get("best_snr"),
                "best_detection_confidence": result.get("best_detection_confidence"),
                "best_novelty_score": result.get("best_novelty_score"),
                "best_depth_ppm": result.get("best_depth_ppm"),
                "best_duration_hours": result.get("best_duration_hours"),
                "best_transit_count": result.get("best_transit_count"),
                "provenance_score": result.get("provenance_score"),
                "signals": result.get("signals", []),
                "priority_score": result.get("priority_score"),
                "pipeline": result.get("pipeline"),
                "exptime": result.get("exptime"),
                "error_message": result.get("error_message"),
            }
            self._data["entries"][str(tic_id)] = entry
            self._data.setdefault("active", {}).pop(str(tic_id), None)
            self._data["last_updated"] = datetime.now(UTC).isoformat()
            self._flush()

    def summary(self) -> dict[str, int]:
        """Return a dict of status → count plus a ``"total"`` key."""
        counts: dict[str, int] = {
            "candidate_found": 0,
            "scanned_clear": 0,
            "no_data": 0,
            "error": 0,
            "active": 0,
            "total": 0,
        }
        with self._lock:
            counts["active"] = len(self._data.get("active", {}))
            for entry in self._data["entries"].values():
                status = entry.get("status", "error")
                if status in counts:
                    counts[status] += 1
                counts["total"] += 1
        return counts

    def _flush(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w") as fh:
                json.dump(self._data, fh, indent=2)
            tmp.replace(self._path)


# ---------------------------------------------------------------------------
# Exclusion lists: TOI, CTOI, confirmed planets
# ---------------------------------------------------------------------------


def _load_prior_discovery_tic_ids(
    log_dir: Path,
    *,
    strict: bool = False,
) -> set[int]:
    """Union completed target IDs from every historical discovery-run log."""
    tic_ids: set[int] = set()
    for path in sorted(log_dir.glob("discovery_run*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("entries must be an object")
            tic_ids.update(int(value) for value in entries)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if strict:
                raise RuntimeError(
                    f"Cannot parse historical discovery log {path}"
                ) from exc
    return tic_ids


def _load_toi_tic_ids(*, strict: bool = False) -> set[int]:
    """Download the TESS TOI table and return the set of TIC IDs it contains."""
    import pandas as pd

    df = pd.read_csv(io.StringIO(_url_text(_EXOFOP_URL)), comment="#")
    tic_col = next(
        (c for c in df.columns if "tic" in c.lower() and "id" in c.lower()),
        None,
    )
    if tic_col is None:
        # Fall back: any column with "tic"
        tic_col = next((c for c in df.columns if "tic" in c.lower()), None)
    if tic_col is None:
        if strict:
            raise RuntimeError("TOI source does not contain a TIC ID column")
        return set()
    result = {int(v) for v in df[tic_col].dropna()}
    if strict and not result:
        raise RuntimeError("TOI source returned no TIC IDs")
    return result


def _load_ctoi_tic_ids(*, strict: bool = False) -> set[int]:
    """Download the ExoFOP CTOI table and return the set of TIC IDs it contains.

    Uses ``Skills/fetch_exofop_ctoi.py`` with its injectable fetch function.
    Returns an empty set on any failure so a normal scan can continue. Strict
    metadata preparation propagates the failure instead.
    """
    import importlib.util

    ctoi_path = _SKILLS_DIR / "fetch_exofop_ctoi.py"
    spec = importlib.util.spec_from_file_location("fetch_exofop_ctoi", ctoi_path)
    if spec is None or spec.loader is None:
        if strict:
            raise RuntimeError("Cannot load fetch_exofop_ctoi.py")
        return set()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        # Exclusion must include every current CTOI. The live ExoFOP export no
        # longer carries the historical ratings column, so min_ratings=1 would
        # silently filter the entire valid table.
        result = module.fetch_ctoi_table(min_ratings=0)
        if strict and result.flag != "OK":
            raise RuntimeError(f"CTOI source returned flag={result.flag}")
        return {int(r["tic_id"]) for r in result.rows if r.get("tic_id")}
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return set()


def _load_confirmed_host_tic_ids(*, strict: bool = False) -> frozenset[int]:
    """Return TIC IDs of confirmed transiting planet hosts from NASA Exoplanet Archive.

    Uses ``Skills/fetch_confirmed_hosts.py``. Returns an empty frozenset on
    failure unless strict metadata preparation requests fail-closed behavior.
    """
    import importlib.util

    host_path = _SKILLS_DIR / "fetch_confirmed_hosts.py"
    spec = importlib.util.spec_from_file_location("fetch_confirmed_hosts", host_path)
    if spec is None or spec.loader is None:
        if strict:
            raise RuntimeError("Cannot load fetch_confirmed_hosts.py")
        return frozenset()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        return module.fetch_confirmed_host_tic_ids(strict=strict)
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()


_ASASSN_CONTRACT_PATH = (
    _REPO_ROOT / "metadata" / "asassn_variability_label_source_contract_v1.json"
)


def _load_asassn_variable_tic_ids(
    candidate_tic_ids: Any,
    *,
    strict: bool = False,
) -> frozenset[int]:
    """Return the subset of *candidate_tic_ids* already flagged as known
    variable stars in the pinned ASAS-SN Catalog X source.

    Uses ``Skills/preflight_tess_asassn_labels.py``'s already-reviewed
    contract loader and exact-TIC VizieR TAP query builder (live, zero
    payload bytes downloaded). A star already classified as a variable
    (eclipsing binary, pulsator, etc.) is a poor novel-transit-search
    target for the same reason a known planet host is — it isn't "unlooked
    at," it's already characterized, just not by a planet-search pipeline.

    Fails open (empty frozenset) on any error unless ``strict=True``, which
    is for immutable metadata preparation only, matching the other
    exclusion-set loaders in this module.
    """
    if not candidate_tic_ids:
        return frozenset()

    import importlib.util

    preflight_path = _SKILLS_DIR / "preflight_tess_asassn_labels.py"
    spec = importlib.util.spec_from_file_location(
        "preflight_tess_asassn_labels", preflight_path
    )
    if spec is None or spec.loader is None:
        if strict:
            raise RuntimeError("Cannot load preflight_tess_asassn_labels.py")
        return frozenset()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        contract = module.load_contract(_ASASSN_CONTRACT_PATH)
        batch_size = int(contract["query"]["batch_size"])
        ids = list(dict.fromkeys(int(t) for t in candidate_tic_ids))
        matched: set[int] = set()
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            query = module.build_query(contract, batch)
            response = module._default_tap(str(contract["catalog"]["tap_endpoint"]), query)
            names, rows = module._response_rows(response)
            tic_idx = names.index("TIC")
            for row in rows:
                raw = str(row[tic_idx]).strip()
                digits = raw[4:].strip() if raw.upper().startswith("TIC ") else raw
                try:
                    matched.add(int(digits))
                except ValueError:
                    continue
        return frozenset(matched)
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _query_one_tile(
    tile_idx: int,
    ra_deg: float,
    dec_deg: float,
    *,
    query_radius_deg: float,
    retry_attempts: int,
    retry_delay: float,
    query_timeout_seconds: float = 30.0,
) -> tuple[Any | None, list[str]]:
    """Query one TIC sky tile with retry. Returns (result_table_or_None, errors).

    ``astroquery.mast``'s own default request timeout is 600 seconds and is
    never overridden elsewhere in this project. A single slow/stalled tile
    query under that default can stall an entire ``full_sweep`` (up to 126
    concurrent tile queries) for up to ten minutes waiting on one straggler.
    Bound it explicitly instead; a real stall now fails fast and retries
    rather than silently blocking the whole sweep.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs
    from astroquery.mast import conf as mast_conf

    # astropy's ConfigItem for astroquery.mast's timeout is int-typed; a
    # float (even a whole-number one like 30.0) raises TypeError.
    mast_conf.timeout = int(query_timeout_seconds)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    errors: list[str] = []
    for attempt in range(1, retry_attempts + 1):
        try:
            result = Catalogs.query_region(
                coord,
                radius=query_radius_deg * u.deg,
                catalog="TIC",
            )
            return result, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"tile {tile_idx} ra={ra_deg:.1f} dec={dec_deg:.1f} "
                f"attempt {attempt}/{retry_attempts}: {exc}"
            )
            if attempt < retry_attempts:
                time.sleep(retry_delay)
    return None, errors


def select_targets(
    n: int = 100,
    tmag_range: tuple[float, float] = (12.0, 14.5),
    exclude_tic_ids: set[int] | None = None,
    *,
    query_radius_deg: float = 0.5,
    max_tiles: int = 126,
    retry_attempts: int = 2,
    retry_delay: float = 2.0,
    full_sweep: bool = False,
    max_workers: int = 6,
    search_log: dict[str, Any] | None = None,
    query_timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Query the TIC catalog and return up to *n* stars ranked by priority.

    Stars in *exclude_tic_ids* (e.g. known TOIs or already-scanned stars) are
    removed before ranking.  The query is intentionally tiled by sky position:
    all-sky TIC magnitude queries are too large for MAST and can be closed by
    the remote service before a response is returned.

    By default (``full_sweep=False``), this stops querying tiles as soon as a
    small buffer (``3n`` candidates) has been collected, for fast incremental
    use. This means the ranked result only reflects whichever few tiles the
    query happened to reach first, in a fixed order — **not** a search across
    the full configured grid. Pass ``full_sweep=True`` to query every tile up
    to *max_tiles* (in parallel, via *max_workers* threads) before ranking,
    so "top n" genuinely means top-n across the whole swept area. This is
    still only a sample of the grid defined by ``_DEFAULT_SEARCH_CENTERS``
    (a fixed set of cone-search tiles), not an exhaustive survey of the sky
    — see ``search_log`` for the exact coverage achieved.

    Args:
        n: Maximum number of targets to return.
        tmag_range: ``(min_tmag, max_tmag)`` magnitude range for the query.
        exclude_tic_ids: TIC IDs to omit from the results.
        query_radius_deg: Cone-search radius for each TIC tile.
        max_tiles: Maximum number of sky tiles to query.
        retry_attempts: Attempts per tile before skipping it.
        retry_delay: Seconds to wait between retry attempts.
        full_sweep: If ``True``, query every tile up to *max_tiles* (in
            parallel) before ranking, instead of stopping early. Use this
            for an actual wide/optimized search; the default fast/early-stop
            behavior is preserved for existing incremental callers.
        max_workers: Thread pool size when ``full_sweep=True``.
        search_log: Optional mutable dict populated in-place with exactly
            what was searched: ``tiles_configured``, ``tiles_queried``,
            ``tiles_failed``, ``tile_errors``, ``sky_coverage_deg2``,
            ``raw_candidates_before_exclusion``, ``excluded_count``,
            ``full_sweep``, ``elapsed_seconds``. Populated regardless of
            ``full_sweep``, so callers can always see the actual search
            extent rather than assuming a wide search occurred.
        query_timeout_seconds: Per-tile-query request timeout, overriding
            ``astroquery.mast``'s own 600-second default. A slow/stalled
            single tile under that default can stall an entire
            ``full_sweep`` (up to *max_tiles* concurrent queries) for up to
            ten minutes waiting on one straggler; bounding it here means a
            real stall fails fast and retries instead.

    Returns:
        List of dicts, sorted by ``"priority"`` descending, each with TIC ID,
        coordinates, stellar metadata, contamination, radius, and priority.
    """
    exclude = exclude_tic_ids or set()
    min_tmag, max_tmag = tmag_range
    target_pool_size = max(n * 3, n)

    targets: list[dict[str, Any]] = []
    seen: set[int] = set()
    tile_errors: list[str] = []
    tiles_queried = 0
    tiles_failed = 0
    raw_rows_matching_query = 0
    _search_start = time.monotonic()

    tiles = list(enumerate(_DEFAULT_SEARCH_CENTERS[:max_tiles], 1))

    def _process_tile_result(result: Any) -> None:
        nonlocal raw_rows_matching_query
        for row in result:
            obj_type = _row_get(row, "objType")
            if obj_type is not None and str(obj_type).strip().upper() != "STAR":
                continue
            try:
                tic_id = int(_row_get(row, "ID"))
            except (ValueError, TypeError):
                continue

            try:
                tmag = float(_row_get(row, "Tmag"))
            except (ValueError, TypeError):
                continue
            if tmag != tmag or not (min_tmag <= tmag <= max_tmag):  # NaN guard
                continue

            # Counts every row matching the query's own structural filters
            # (star type, magnitude range), independent of exclusion —
            # i.e. how many real candidates existed before we removed
            # already-known/already-scanned ones.
            if tic_id not in seen:
                raw_rows_matching_query += 1

            if tic_id in exclude or tic_id in seen:
                continue

            try:
                raw_teff = _row_get(row, "Teff")
                teff: float | None = float(raw_teff) if raw_teff is not None else None
            except (ValueError, TypeError):
                teff = None

            try:
                raw_contratio = _row_get(row, "contratio")
                contratio: float | None = (
                    float(raw_contratio) if raw_contratio is not None else None
                )
            except (ValueError, TypeError):
                contratio = None

            try:
                raw_rad = _row_get(row, "rad")
                radius_rsun: float | None = float(raw_rad) if raw_rad is not None else None
            except (ValueError, TypeError):
                radius_rsun = None

            pri = priority_score(
                tmag, teff=teff, contratio=contratio, radius_rsun=radius_rsun
            )
            seen.add(tic_id)
            targets.append(
                {
                    "tic_id": tic_id,
                    "ra_deg": _row_float_or_none(row, "ra"),
                    "dec_deg": _row_float_or_none(row, "dec"),
                    "tmag": tmag,
                    "teff": teff,
                    "contratio": contratio,
                    "radius_rsun": radius_rsun,
                    "priority": pri,
                }
            )

    if full_sweep:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _query_one_tile,
                    tile_idx,
                    ra_deg,
                    dec_deg,
                    query_radius_deg=query_radius_deg,
                    retry_attempts=retry_attempts,
                    retry_delay=retry_delay,
                    query_timeout_seconds=query_timeout_seconds,
                ): (tile_idx, ra_deg, dec_deg)
                for tile_idx, (ra_deg, dec_deg) in tiles
            }
            for fut in as_completed(futures):
                tiles_queried += 1
                result, errors = fut.result()
                tile_errors.extend(errors)
                if result is None:
                    tiles_failed += 1
                    continue
                _process_tile_result(result)
    else:
        for tile_idx, (ra_deg, dec_deg) in tiles:
            tiles_queried += 1
            result, errors = _query_one_tile(
                tile_idx, ra_deg, dec_deg,
                query_radius_deg=query_radius_deg,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
                query_timeout_seconds=query_timeout_seconds,
            )
            tile_errors.extend(errors)
            if result is None:
                tiles_failed += 1
                continue
            _process_tile_result(result)
            if len(targets) >= target_pool_size:
                break

    if not targets and tile_errors:
        sample = "; ".join(tile_errors[-3:])
        raise RuntimeError(
            f"TIC target selection failed across {len(tile_errors)} attempts: {sample}"
        )

    targets.sort(key=lambda t: t["priority"], reverse=True)

    if search_log is not None:
        search_log.update(
            {
                "tiles_configured": len(tiles),
                "tiles_queried": tiles_queried,
                "tiles_failed": tiles_failed,
                "tile_errors": list(tile_errors),
                "sky_coverage_deg2": round(
                    tiles_queried * math.pi * query_radius_deg**2, 4
                ),
                "raw_candidates_before_exclusion": raw_rows_matching_query,
                "candidates_after_exclusion": len(targets),
                "excluded_count": len(exclude),
                "full_sweep": full_sweep,
                "elapsed_seconds": round(time.monotonic() - _search_start, 3),
            }
        )

    return targets[:n]


def _row_float_or_none(row: Any, key: str) -> float | None:
    """Return one finite catalog float, or ``None`` for missing/bad cells."""
    try:
        value = float(_row_get(row, key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def inspect_target_products(
    target: dict[str, Any],
    *,
    mission: str,
    pipeline: str,
    exptime: str,
    search_fn: Any = None,
) -> dict[str, Any]:
    """Attach exact MAST light-curve product metadata without downloading files."""
    if search_fn is None:
        import lightkurve as lk

        search_fn = lk.search_lightcurve

    target_id = f"TIC {int(target['tic_id'])}"
    search = search_fn(
        target_id,
        mission=mission,
        author=pipeline,
        exptime=exptime,
    )
    table = search.table
    colnames = set(getattr(table, "colnames", ()))
    uri_column = "dataURI" if "dataURI" in colnames else "dataURL"
    products: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    for row in table:
        uri = str(_row_get(row, uri_column) or "")
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        size_raw = _row_get(row, "size")
        try:
            size_bytes: int | None = max(0, int(float(size_raw)))
        except (TypeError, ValueError):
            size_bytes = None
        sector_raw = _row_get(row, "sequence_number")
        try:
            sector = int(sector_raw)
        except (TypeError, ValueError):
            sector = None
        products.append(
            {
                "uri": uri,
                "filename": str(_row_get(row, "productFilename") or Path(uri).name),
                "size_bytes": size_bytes,
                "sector": sector,
            }
        )

    sectors = sorted({item["sector"] for item in products if item["sector"] is not None})
    result = dict(target)
    result.update(
        {
            "products": tuple(products),
            "product_count": len(products),
            "total_bytes": sum(
                int(item["size_bytes"] or 0) for item in products
            ),
            "sectors": tuple(sectors),
            "n_sectors": len(sectors),
        }
    )
    result["priority"] = priority_score(
        float(result["tmag"]),
        teff=result.get("teff"),
        n_sectors=result["n_sectors"],
        contratio=result.get("contratio"),
        radius_rsun=result.get("radius_rsun"),
    )
    return result


def _storage_penalty(estimated_download_gb: float) -> int:
    if estimated_download_gb <= 5.0:
        return 0
    if estimated_download_gb <= 25.0:
        return 1
    if estimated_download_gb <= 100.0:
        return 2
    if estimated_download_gb <= 250.0:
        return 3
    return 5


def _queue_row(target: dict[str, Any], *, pipeline: str) -> dict[str, Any]:
    """Map verified metadata to the mandatory live-search queue contract."""
    estimated_gb = float(target["total_bytes"]) / 1_000_000_000.0
    penalty = _storage_penalty(estimated_gb)
    tmag = float(target["tmag"])
    teff = target.get("teff")
    scientific_novelty = 3.0
    prior_significance = 2.0 if teff is not None and float(teff) <= 5500 else 1.0
    followup_leverage = 3.0 if tmag <= 13.0 else 2.0 if tmag <= 14.0 else 1.0
    data_quality = round(3.0 * float(target["priority"]), 3)
    method_advantage = 2.0
    publication_value = 2.0
    community_integration = 3.0
    new_followup_balance = 3.0
    total_priority = round(
        scientific_novelty
        + prior_significance
        + followup_leverage
        + data_quality
        + method_advantage
        + publication_value
        + community_integration
        + new_followup_balance
        - penalty,
        3,
    )
    sectors = ",".join(str(value) for value in target["sectors"])
    return {
        "target_id": f"TIC {int(target['tic_id'])}",
        "project": "2026 Exoplanet Research",
        "source": "MAST TIC and TESS light-curve product metadata",
        "catalog_ids": f"TIC {int(target['tic_id'])}",
        "ra_deg": target.get("ra_deg"),
        "dec_deg": target.get("dec_deg"),
        "data_products_available": (
            f"{pipeline} light curves: {target['product_count']}; sectors: {sectors}"
        ),
        "estimated_download_gb": round(estimated_gb, 9),
        "search_category": "new_target",
        "scientific_novelty": scientific_novelty,
        "prior_significance": prior_significance,
        "followup_leverage": followup_leverage,
        "data_quality": data_quality,
        "method_advantage": method_advantage,
        "publication_value": publication_value,
        "community_integration": community_integration,
        "new_followup_balance": new_followup_balance,
        "storage_cost_penalty": penalty,
        "total_priority": total_priority,
        "status": (
            "queued"
            if target["product_count"] and total_priority >= 18.0
            else "rejected_no_products"
            if not target["product_count"]
            else "rejected_low_priority"
        ),
        "notes": (
            "Selected from the faint-star novelty frontier after TOI, CTOI, "
            "confirmed-host, and prior-scan exclusions; exact MAST products verified."
        ),
        "citations": (
            "https://mast.stsci.edu/; "
            "https://exoplanetarchive.ipac.caltech.edu/; "
            "https://exofop.ipac.caltech.edu/tess/"
        ),
    }


def prepare_live_search_snapshot(
    targets: list[dict[str, Any]],
    *,
    queue_path: Path,
    immutable_snapshot_path: Path,
    batch_manifest_path: Path,
    dataset_manifest_path: Path,
    repo_root: Path,
    mission: str = "TESS",
    pipeline: str = "QLP",
    exptime: str = "long",
    workers: int = 6,
    inspector_fn: Any = None,
    batch_id: str = "tess_live_search_v1",
    dataset_id: str = "tess_live_search_v1",
    replace: bool = False,
) -> dict[str, Any]:
    """Verify target products and freeze queue/batch/dataset metadata only."""
    outputs = (
        queue_path,
        immutable_snapshot_path,
        batch_manifest_path,
        dataset_manifest_path,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not replace:
        raise FileExistsError(
            "Refusing to overwrite live-search preparation evidence: "
            + ", ".join(existing)
        )
    if not targets:
        raise RuntimeError("Cannot prepare a live-search snapshot from zero targets")

    inspect = inspector_fn or inspect_target_products
    worker_count = max(1, min(workers, len(targets)))
    print(
        f"Preparing live-search metadata for {len(targets)} targets "
        f"(workers={worker_count}; no raw downloads) …",
        flush=True,
    )
    started = time.monotonic()
    inspected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                inspect,
                target,
                mission=mission,
                pipeline=pipeline,
                exptime=exptime,
            )
            for target in targets
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            inspected.append(result)
            print(
                f"  [{index}/{len(targets)}] TIC {result['tic_id']} "
                f"products={result['product_count']} "
                f"{_progress_suffix(index, len(targets), started)}",
                flush=True,
            )

    if any(
        target.get("ra_deg") is None
        or target.get("dec_deg") is None
        or any(
            product["size_bytes"] is None or int(product["size_bytes"]) <= 0
            for product in target["products"]
        )
        for target in inspected
    ):
        raise RuntimeError(
            "Cannot freeze live-search metadata with missing coordinates or product sizes"
        )

    rows = [_queue_row(target, pipeline=pipeline) for target in inspected]
    rows.sort(key=lambda row: (-float(row["total_priority"]), str(row["target_id"])))
    queued_rows = [row for row in rows if row["status"] == "queued"]
    if not queued_rows:
        raise RuntimeError(
            "No product-backed targets satisfy the normal live-search priority gate (>=18)"
        )
    fieldnames = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(queue_path, buffer.getvalue())

    snapshot_buffer = io.StringIO()
    snapshot_writer = csv.DictWriter(
        snapshot_buffer, fieldnames=fieldnames, lineterminator="\n"
    )
    snapshot_writer.writeheader()
    snapshot_writer.writerows(queued_rows)
    _atomic_write_text(immutable_snapshot_path, snapshot_buffer.getvalue())

    root = repo_root.resolve()
    queue_relative = queue_path.resolve().relative_to(root).as_posix()
    snapshot_relative = immutable_snapshot_path.resolve().relative_to(root).as_posix()
    snapshot_sha256 = sha256_file(immutable_snapshot_path)
    queued_target_ids = {str(row["target_id"]) for row in queued_rows}
    queued_targets = [
        target
        for target in inspected
        if f"TIC {int(target['tic_id'])}" in queued_target_ids
    ]
    total_bytes = sum(int(target["total_bytes"]) for target in queued_targets)
    free_gb = shutil.disk_usage(root).free / 1_000_000_000.0
    managed_paths = (
        "data",
        "datasets",
        "cache",
        ".cache",
        "artifacts",
        "outputs",
        "downloads",
        "tmp",
    )
    managed_bytes = sum(
        path.stat().st_size
        for name in managed_paths
        for path in (root / name).rglob("*")
        if path.is_file()
    )
    managed_gb = managed_bytes / 1_000_000_000.0
    remaining_project_cap_gb = max(0.0, 100.0 - managed_gb)
    estimated_download_gb = total_bytes / 1_000_000_000.0
    if estimated_download_gb > remaining_project_cap_gb:
        raise RuntimeError(
            f"Prepared batch estimate {estimated_download_gb:.3f} GB exceeds "
            f"remaining 100 GB project-data headroom {remaining_project_cap_gb:.3f} GB"
        )
    created_at = datetime.now(UTC)
    batch_manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "project": "2026 Exoplanet Research",
        "role": "live_search",
        "acquisition_mode": "stream_process_evict",
        "target_queue": queue_relative,
        "target_queue_snapshot": {
            "path": snapshot_relative,
            "sha256": snapshot_sha256,
        },
        "source_archive": "MAST TIC catalog and TESS light-curve products",
        "query": {
            "mission": mission,
            "pipeline": pipeline,
            "exptime": exptime,
            "target_count": len(queued_rows),
            "selection_rubric": (
                "eight policy criteria scored 0-3; product-backed targets require >=18"
            ),
        },
        "estimated_download_gb": round(estimated_download_gb, 9),
        "max_allowed_download_gb": round(remaining_project_cap_gb, 9),
        "project_managed_data_before_gb": round(managed_gb, 9),
        "expected_raw_files": sum(
            int(target["product_count"]) for target in queued_targets
        ),
        "expected_derived_gb": 0.0,
        "free_space_before_gb": round(free_gb, 3),
        "free_space_required_after_gb": 10.0,
        "eviction_rule": "evict redownloadable raw products after candidate rows are written",
        "pin_rule": "pin queue, manifests, ledger rows, and unresolved candidate evidence",
        "stop_condition": (
            "stop before 100 GB project-managed data, below 10 GB free, metadata "
            "mismatch, duplicate target, or elevated MAST failure/throttling rate"
        ),
        "manifest_owner": "Exoplanet agent",
        "created_at": created_at.isoformat(),
        "product_inventory": [
            {
                "target_id": f"TIC {int(target['tic_id'])}",
                "products": list(target["products"]),
            }
            for target in sorted(queued_targets, key=lambda item: int(item["tic_id"]))
        ],
    }
    _atomic_write_text(
        batch_manifest_path,
        json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
    )

    dataset_manifest = DatasetManifest(
        schema_version=1,
        dataset_id=dataset_id,
        project="2026 Exoplanet Research",
        role="live_search",
        source_name="MAST TIC catalog and TESS light-curve product metadata",
        source_url="https://mast.stsci.edu/",
        instrument="TESS cameras",
        target_ids={
            "namespace": "TIC",
            "count": len(queued_rows),
            "selection": "faint-star novelty frontier with known-object exclusions",
        },
        time_range={
            "status": "deferred",
            "reason": (
                "per-target sector coverage is in the queue and exact product URIs "
                "are in the batch manifest inventory"
            ),
        },
        cadence={"status": "known", "value": exptime},
        band_or_frequency="TESS optical bandpass, approximately 600-1000 nm",
        data_product_type="checksummed live-search target queue with MAST product inventory",
        acquired_at=created_at,
        local_path=snapshot_relative,
        sha256=snapshot_sha256,
        license="NASA/STScI public archive data; acknowledge MAST and the TESS mission",
        label_source="unlabeled live search after TOI, CTOI, and confirmed-host exclusions",
        label_confidence="unlabeled",
        preprocessing_version="star-scanner-live-search-queue-v1",
        known_caveats=(
            "Product sizes are MAST metadata estimates and may differ from cached bytes.",
            "The queue contains light curves only; target-pixel products require "
            "a separate follow-up manifest.",
        ),
        row_count=len(queued_rows),
        group_count=len(queued_rows),
    )
    _atomic_write_text(
        dataset_manifest_path,
        json.dumps(dataset_manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
    )
    validation = validate_dataset_manifest(
        dataset_manifest_path,
        repo_root=root,
    )
    if not validation.ok:
        raise RuntimeError(
            "Written live-search dataset manifest failed validation: "
            + "; ".join(validation.errors)
        )
    print(
        f"Prepared {len(queued_rows)} queued targets / "
        f"{batch_manifest['expected_raw_files']} products "
        f"/ {batch_manifest['estimated_download_gb']:.6f} GB (metadata only)",
        flush=True,
    )
    return {
        "queue_path": str(queue_path),
        "immutable_snapshot_path": str(immutable_snapshot_path),
        "batch_manifest_path": str(batch_manifest_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "target_count": len(queued_rows),
        "inspected_count": len(inspected),
        "rejected_count": len(inspected) - len(queued_rows),
        "expected_raw_files": batch_manifest["expected_raw_files"],
        "estimated_download_gb": batch_manifest["estimated_download_gb"],
    }


@dataclass(frozen=True)
class PreparedLiveSearchBundle:
    """Validated immutable live-search membership and raw-product scope."""

    dataset_id: str
    batch_id: str
    targets: tuple[dict[str, Any], ...]


def _shard_scoped_path(path: Path, shard_index: int, shard_count: int) -> Path:
    """Return a collision-free per-shard path when process sharding is active."""
    if shard_count <= 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard_index}of{shard_count}{path.suffix}")


def load_prepared_live_search_bundle(
    *,
    dataset_manifest_path: Path,
    batch_manifest_path: Path,
    repo_root: Path,
) -> PreparedLiveSearchBundle:
    """Load and cross-check the frozen queue, manifest, and product inventory."""
    root = repo_root.resolve()
    validation = validate_dataset_manifest(dataset_manifest_path, repo_root=root)
    if not validation.ok:
        raise RuntimeError(
            "Live-search dataset manifest failed validation: "
            + "; ".join(validation.errors)
        )
    manifest = load_dataset_manifest(dataset_manifest_path)
    if manifest.role != "live_search":
        raise RuntimeError(f"Dataset role must be live_search, got {manifest.role}")

    try:
        batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read batch manifest {batch_manifest_path}") from exc
    snapshot = batch.get("target_queue_snapshot", {})
    if batch.get("role") != "live_search" or batch.get("batch_id") != manifest.dataset_id:
        raise RuntimeError("Batch identity/role does not match the dataset manifest")
    if snapshot.get("path") != manifest.local_path or snapshot.get("sha256") != manifest.sha256:
        raise RuntimeError("Batch snapshot path/checksum does not match DatasetManifest")

    queue_path = (root / manifest.local_path).resolve()
    try:
        queue_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Prepared queue escapes the repository root") from exc
    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != manifest.row_count or not rows:
        raise RuntimeError("Prepared queue row count does not match DatasetManifest")

    inventory: dict[str, tuple[str, ...]] = {}
    for item in batch.get("product_inventory", []):
        target_id = str(item.get("target_id", ""))
        if not target_id or target_id in inventory:
            raise RuntimeError("Batch product inventory has a missing/duplicate target")
        products = item.get("products", [])
        uris = tuple(str(product.get("uri", "")) for product in products)
        if not uris or any(not uri for uri in uris):
            raise RuntimeError(f"Batch target {target_id} has incomplete raw URIs")
        if any(int(product.get("size_bytes") or 0) <= 0 for product in products):
            raise RuntimeError(f"Batch target {target_id} has invalid product sizes")
        inventory[target_id] = uris

    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        target_id = str(row.get("target_id", ""))
        if row.get("status") != "queued" or target_id in seen:
            raise RuntimeError("Prepared queue contains a non-queued or duplicate target")
        try:
            tic_id = int(target_id.removeprefix("TIC "))
            priority = float(row["total_priority"])
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Invalid prepared target row: {target_id!r}") from exc
        if target_id not in inventory:
            raise RuntimeError(f"Prepared target {target_id} has no product inventory")
        seen.add(target_id)
        targets.append(
            {
                "tic_id": tic_id,
                "target_id": target_id,
                "priority": priority,
                "raw_uris": inventory[target_id],
            }
        )
    if seen != set(inventory):
        raise RuntimeError("Queue membership and product inventory targets differ")
    if int(batch.get("query", {}).get("target_count", -1)) != len(targets):
        raise RuntimeError("Batch target count does not match queue membership")
    return PreparedLiveSearchBundle(
        dataset_id=manifest.dataset_id,
        batch_id=str(batch["batch_id"]),
        targets=tuple(targets),
    )


def _time_window(provenance: dict[str, Any]) -> str:
    values = tuple(int(value) for value in provenance.get("sectors_or_quarters", ()))
    if not values:
        raise RuntimeError("Fetch provenance has no sectors/quarters")
    return "sectors " + ",".join(str(value) for value in values)


def _ledger_record_for_outcome(
    *,
    target: dict[str, Any],
    source_dataset_id: str,
    row: dict[str, Any] | None,
    pipeline_context: dict[str, Any],
    min_snr: float,
    max_peaks: int,
    max_period_grid_points: int | None,
    scorer: str,
    pipeline: str,
    exptime: str,
    model_path: Path | None,
    failure_message: str | None = None,
) -> CandidateLedgerRecord:
    """Build one schema-v2 record after verifying fetched products exactly."""
    provenance = (
        row.get("fetch_provenance", {})
        if row is not None
        else pipeline_context.get("fetch_provenance", {})
    )
    fetched_uris = tuple(str(uri) for uri in provenance.get("raw_uris", ()))
    expected_uris = tuple(str(uri) for uri in target["raw_uris"])
    if fetched_uris != expected_uris:
        raise RuntimeError(
            f"Fetched product URIs for {target['target_id']} differ from frozen inventory"
        )
    generator_params: dict[str, Any] = {
        "period_min_days": 0.5,
        "period_max_days": 500.0,
        "duration_min_hours": 0.5,
        "duration_max_hours": 12.0,
        "n_durations": 20,
        "min_snr": min_snr,
        "max_peaks": max_peaks,
        "max_period_grid_points": max_period_grid_points,
        "pipeline": pipeline,
        "exptime": exptime,
    }
    target_id = str(target["target_id"])
    if row is None:
        candidate_id = f"{target_id.replace(' ', '_')}_null"
        model_scores: dict[str, float | None] = {}
        calibrated_scores: dict[str, float | None] = {}
        review_status = "preprocessing_failure" if failure_message else "unreviewed"
        review_notes = failure_message or "No BLS signal exceeded the configured gate"
    else:
        candidate_id = str(row["candidate_id"])
        posterior = row.get("posterior", {})
        scores = row.get("scores", {})
        model_scores = {
            **{f"posterior_{key}": value for key, value in posterior.items()},
            **{str(key): value for key, value in scores.items()},
        }
        for key in (
            "xgb_planet_probability",
            "ensemble_planet_probability",
            "cnn_planet_probability",
            "full_ensemble_planet_probability",
        ):
            if key in row:
                model_scores[key] = row[key]
        calibrated_scores = {
            str(key): value for key, value in row.get("calibrated_posterior", {}).items()
        }
        if "false_discovery_estimate" in row:
            model_scores["false_discovery_estimate"] = row["false_discovery_estimate"]
            model_scores["false_discovery_reference_n"] = row.get(
                "false_discovery_reference_n"
            )
            model_scores["false_discovery_reference_negatives"] = row.get(
                "false_discovery_reference_negatives"
            )
        review_status = "unreviewed"
        review_notes = ""
    model_versions = {"exo_toolkit": __version__, "scorer": scorer}
    if model_path is not None:
        model_versions["model_path"] = str(model_path)
        model_versions["model_sha256"] = sha256_file(model_path)
    if row is not None and row.get("calibration_dataset_id"):
        model_versions["calibration_dataset_id"] = str(row["calibration_dataset_id"])
        model_versions["threshold_version"] = str(row["threshold_version"])
        model_versions["candidate_context_id"] = str(row["candidate_context_id"])
    regeneration_command = (
        f"caffeinate -i .venv/bin/python Skills/star_scanner.py --target "
        f"{int(target['tic_id'])} --mission TESS --pipeline {pipeline} "
        f"--exptime {exptime} --min-snr {min_snr} --max-peaks {max_peaks} "
        f"--max-period-grid-points {max_period_grid_points or 0} --scorer {scorer}"
    )
    if model_path is not None:
        regeneration_command += f" --model-path {model_path}"
    return CandidateLedgerRecord(
        schema_version=2,
        candidate_id=candidate_id,
        source_dataset_id=source_dataset_id,
        target_id=target_id,
        mission="TESS",
        time_window=_time_window(provenance),
        raw_uris=fetched_uris,
        preprocess_version=str(
            pipeline_context.get(
                "preprocess_version", f"exo-toolkit-{__version__}:clean_lightcurve"
            )
        ),
        candidate_generator="astropy.timeseries.BoxLeastSquares",
        candidate_generator_params=generator_params,
        model_versions=model_versions,
        model_scores=model_scores,
        calibrated_scores=calibrated_scores,
        score_quantiles=(
            {"full_ensemble_planet_probability": row.get("score_quantile")}
            if row is not None and "score_quantile" in row
            else {}
        ),
        injection_context={"status": "not_injected", "source_role": "live_search"},
        nearest_known_artifacts=(),
        review_status=review_status,
        review_notes=review_notes,
        regeneration_command=regeneration_command,
    )


# ---------------------------------------------------------------------------
# Single-star scan
# ---------------------------------------------------------------------------


def scan_star(
    tic_id: int,
    *,
    mission: str = "TESS",
    log: ScanLog | None = None,
    min_snr: float = 5.0,
    max_peaks: int = 5,
    max_period_grid_points: int | None = 20_000,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    pipeline: str | None = None,
    exptime: str | None = None,
    priority: float | None = None,
    capture_ledger_context: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline on one star and optionally persist the result.

    Args:
        tic_id: Numeric TIC identifier.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        log: Optional :class:`ScanLog`; when provided the result is recorded.
        min_snr: Minimum BLS SNR threshold.
        max_peaks: Maximum signals to search for per star.
        max_period_grid_points: Maximum BLS trial periods per peak.
        scorer: ``"bayesian"``, ``"xgboost"``, or ``"ensemble"``.
        model_path: XGBoost model JSON (required for xgboost/ensemble).
        pipeline: Optional MAST pipeline/author override.
        exptime: Optional MAST exposure hint.
        priority: Pre-computed priority score to store in the log entry.
        capture_ledger_context: Preserve full pipeline rows and fetch provenance
            for immediate schema-v2 ledger insertion by the prepared-batch path.

    Returns:
        Dict with keys: ``status``, ``n_signals``, ``best_period_days``,
        ``best_fpp``, ``best_pathway``, ``priority_score``, ``error_message``.

        ``status`` is one of ``"candidate_found"``, ``"scanned_clear"``,
        ``"no_data"``, or ``"error"``.
    """
    target_id = f"TIC {tic_id}"
    result: dict[str, Any] = {
        "n_signals": 0,
        "best_period_days": None,
        "best_fpp": None,
        "best_pathway": None,
        "best_snr": None,
        "best_detection_confidence": None,
        "best_novelty_score": None,
        "best_depth_ppm": None,
        "best_duration_hours": None,
        "best_transit_count": None,
        "provenance_score": None,
        "signals": [],
        "priority_score": priority,
        "pipeline": pipeline,
        "exptime": exptime,
        "error_message": None,
    }

    pipeline_context: dict[str, Any] = {}
    try:
        rows = run_pipeline(
            target_id,
            mission,  # type: ignore[arg-type]
            min_snr=min_snr,
            max_peaks=max_peaks,
            max_period_grid_points=max_period_grid_points,
            scorer=scorer,
            model_path=model_path,
            pipeline=pipeline,
            exptime=exptime,
            run_context=pipeline_context if capture_ledger_context else None,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        result["status"] = "no_data" if _is_no_data_error(message) else "error"
        result["error_message"] = message
        if capture_ledger_context:
            result["_ledger_rows"] = []
            result["_pipeline_context"] = pipeline_context
        if log is not None:
            log.record(tic_id, result["status"], result)
        return result

    if not rows:
        result["status"] = "scanned_clear"
    else:
        result["status"] = "candidate_found"
        result["n_signals"] = len(rows)
        best = min(rows, key=lambda r: r["scores"]["false_positive_probability"])
        result["signals"] = [_summarize_signal_row(row) for row in rows]
        result["best_period_days"] = best["period_days"]
        result["best_fpp"] = best["scores"]["false_positive_probability"]
        result["best_pathway"] = best["pathway"]
        result["best_snr"] = best.get("snr")
        result["best_detection_confidence"] = best.get("scores", {}).get("detection_confidence")
        result["best_novelty_score"] = best.get("scores", {}).get("novelty_score")
        result["best_depth_ppm"] = best.get("depth_ppm")
        result["best_duration_hours"] = best.get("duration_hours")
        result["best_transit_count"] = best.get("transit_count")
        result["provenance_score"] = best.get("provenance_score")

    if capture_ledger_context:
        result["_ledger_rows"] = rows
        result["_pipeline_context"] = pipeline_context

    if log is not None:
        log.record(tic_id, result["status"], result)
    return result


def _summarize_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return the durable, review-relevant subset of a pipeline output row."""
    scores = row.get("scores", {})
    return {
        "candidate_id": row.get("candidate_id"),
        "period_days": row.get("period_days"),
        "epoch_bjd": row.get("epoch_bjd"),
        "duration_hours": row.get("duration_hours"),
        "depth_ppm": row.get("depth_ppm"),
        "transit_count": row.get("transit_count"),
        "snr": row.get("snr"),
        "false_positive_probability": scores.get("false_positive_probability"),
        "detection_confidence": scores.get("detection_confidence"),
        "novelty_score": scores.get("novelty_score"),
        "provenance_score": row.get("provenance_score"),
        "pathway": row.get("pathway"),
    }


def _is_no_data_error(message: str) -> bool:
    """Return True when a pipeline exception means the target has no usable data."""
    normalized = message.lower()
    no_data_markers = (
        "no tess light curves found",
        "no kepler light curves found",
        "no k2 light curves found",
        "no light curves found",
        "no data found",
    )
    return any(marker in normalized for marker in no_data_markers)


def run_target_scan(
    log_path: Path,
    tic_id: int,
    *,
    mission: str = "TESS",
    min_snr: float = 5.0,
    max_peaks: int = 5,
    max_period_grid_points: int | None = 20_000,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    pipeline: str = "QLP",
    exptime: str = "long",
) -> dict[str, Any]:
    """Scan one explicit TIC target with durable active-state logging."""
    log = ScanLog(log_path)
    start_time = time.monotonic()
    target = {"tic_id": tic_id, "priority": None}
    log.mark_started(tic_id, target, pipeline=pipeline, exptime=exptime)
    print(
        f"[start] TIC {tic_id}  pipeline={pipeline}  exptime={exptime}  "
        f"max_peaks={max_peaks}  period_grid≤{max_period_grid_points or 'auto'}  "
        f"active={log.summary()['active']}  "
        f"{_progress_suffix(0, 1, start_time)}",
        flush=True,
    )
    result = scan_star(
        tic_id,
        mission=mission,
        log=log,
        min_snr=min_snr,
        max_peaks=max_peaks,
        max_period_grid_points=max_period_grid_points,
        scorer=scorer,
        model_path=model_path,
        pipeline=pipeline,
        exptime=exptime,
    )
    print(
        f"[1/1] TIC {tic_id}  {_status_text(result)}  "
        f"active={log.summary()['active']}  "
        f"{_progress_suffix(1, 1, start_time)}",
        flush=True,
    )
    return result


def run_prepared_live_search(
    bundle: PreparedLiveSearchBundle,
    *,
    log_path: Path,
    candidate_db_path: Path,
    workers: int = 6,
    request_delay: float = 0.5,
    min_snr: float = 5.0,
    max_peaks: int = 5,
    max_period_grid_points: int | None = 20_000,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    pipeline: str = "QLP",
    exptime: str = "long",
    shard_index: int = 0,
    shard_count: int = 1,
    heartbeat_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Execute one collision-free shard and append every outcome to ledger v2."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if heartbeat_interval_seconds <= 0.0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    scoped_log = _shard_scoped_path(log_path, shard_index, shard_count)
    scoped_db = _shard_scoped_path(candidate_db_path, shard_index, shard_count)
    shard_targets = [
        target
        for target in bundle.targets
        if int(target["tic_id"]) % shard_count == shard_index
    ]
    log = ScanLog(scoped_log)
    with CandidateDatabase(scoped_db) as database:
        completed_targets = database.completed_provenanced_target_ids(
            bundle.dataset_id
        )
        targets = [
            target
            for target in shard_targets
            if target["target_id"] not in completed_targets
        ]
        print(
            f"Prepared live search {bundle.dataset_id}: shard "
            f"{shard_index}/{shard_count}, batch_total={len(bundle.targets)}, "
            f"shard_total={len(shard_targets)}, pending={len(targets)}, "
            f"workers={min(max(1, workers), max(1, len(targets)))}, "
            f"heartbeat={heartbeat_interval_seconds:.0f}s",
            flush=True,
        )
        if not targets:
            return {
                "mode": "prepared_scan",
                "items_processed": 0,
                "items_written": 0,
                "items_failed": 0,
                "output_paths": (str(scoped_log), str(scoped_db)),
            }

        started = time.monotonic()
        limiter = _StartRateLimiter(request_delay)
        worker_count = min(max(1, workers), len(targets))
        n_done = 0
        n_failed = 0
        n_written = 0
        active: set[int] = set()
        state_lock = threading.Lock()
        print_lock = threading.Lock()
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(heartbeat_interval_seconds):
                with state_lock:
                    completed = n_done
                    active_count = len(active)
                pending = max(0, len(targets) - completed - active_count)
                progress = (
                    _progress_suffix(completed, len(targets), started)
                    if completed
                    else f"elapsed={time.monotonic() - started:.0f}s ETA=pending"
                )
                with print_lock:
                    print(
                        f"  [heartbeat] completed={completed}/{len(targets)} "
                        f"active={active_count} pending={pending} {progress}",
                        flush=True,
                    )

        def scan_target(target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            limiter.wait()
            tic_id = int(target["tic_id"])
            with state_lock:
                active.add(tic_id)
                completed = n_done
                active_count = len(active)
            with print_lock:
                print(
                    f"  [start] {target['target_id']} "
                    f"priority={float(target['priority']):.3f} active={active_count} "
                    f"{_progress_suffix(completed, len(targets), started)}",
                    flush=True,
                )
            result = scan_star(
                tic_id,
                mission="TESS",
                min_snr=min_snr,
                max_peaks=max_peaks,
                max_period_grid_points=max_period_grid_points,
                scorer=scorer,
                model_path=model_path,
                pipeline=pipeline,
                exptime=exptime,
                priority=float(target["priority"]),
                capture_ledger_context=True,
            )
            return target, result

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(scan_target, target) for target in targets]
                for future in as_completed(futures):
                    target, result = future.result()
                    tic_id = int(target["tic_id"])
                    rows = list(result.pop("_ledger_rows", []))
                    pipeline_context = dict(result.pop("_pipeline_context", {}))
                    records: list[CandidateLedgerRecord] = []
                    try:
                        if rows:
                            records = [
                                _ledger_record_for_outcome(
                                    target=target,
                                    source_dataset_id=bundle.dataset_id,
                                    row=row,
                                    pipeline_context=pipeline_context,
                                    min_snr=min_snr,
                                    max_peaks=max_peaks,
                                    max_period_grid_points=max_period_grid_points,
                                    scorer=scorer,
                                    pipeline=pipeline,
                                    exptime=exptime,
                                    model_path=model_path,
                                )
                                for row in rows
                            ]
                        elif pipeline_context.get("fetch_provenance"):
                            records = [
                                _ledger_record_for_outcome(
                                    target=target,
                                    source_dataset_id=bundle.dataset_id,
                                    row=None,
                                    pipeline_context=pipeline_context,
                                    min_snr=min_snr,
                                    max_peaks=max_peaks,
                                    max_period_grid_points=max_period_grid_points,
                                    scorer=scorer,
                                    pipeline=pipeline,
                                    exptime=exptime,
                                    model_path=model_path,
                                    failure_message=(
                                        str(result["error_message"])
                                        if result["status"] == "error"
                                        else None
                                    ),
                                )
                            ]
                        else:
                            raise RuntimeError(
                                "No exact fetch provenance was available for ledger insertion"
                            )
                        database.insert_provenanced_many(records)
                    except Exception as exc:  # noqa: BLE001
                        result["status"] = "error"
                        result["error_message"] = f"Candidate-ledger write refused: {exc}"
                        records = []

                    log.record(tic_id, result["status"], result)
                    with state_lock:
                        active.discard(tic_id)
                        n_done += 1
                        completed = n_done
                    n_written += len(records)
                    if result["status"] == "error":
                        n_failed += 1
                    with print_lock:
                        print(
                            f"  [{completed}/{len(targets)}] {target['target_id']} "
                            f"{_status_text(result)} ledger_rows={len(records)} "
                            f"{_progress_suffix(completed, len(targets), started)}",
                            flush=True,
                        )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=heartbeat_interval_seconds + 1.0)

    return {
        "mode": "prepared_scan",
        "items_processed": n_done,
        "items_written": n_written,
        "items_failed": n_failed,
        "output_paths": (str(scoped_log), str(scoped_db)),
    }


# ---------------------------------------------------------------------------
# Background scan loop
# ---------------------------------------------------------------------------


def run_background_scan(
    log_path: Path,
    *,
    n_targets: int = 500,
    tmag_range: tuple[float, float] = (12.0, 14.5),
    mission: str = "TESS",
    min_snr: float = 5.0,
    max_peaks: int = 5,
    max_period_grid_points: int | None = 20_000,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    pipeline: str = "QLP",
    exptime: str = "long",
    query_radius_deg: float = 0.5,
    max_target_query_tiles: int = 126,
    workers: int = 6,
    request_delay: float = 0.5,
    prepare_only: bool = False,
    queue_path: Path = Path("data_selection/target_priority_queue.csv"),
    immutable_snapshot_path: Path = Path(
        "data_selection/target_priority_queue.tess_live_search_v1.csv"
    ),
    batch_manifest_path: Path = Path(
        "data_selection/batch_manifests/tess_live_search_v1.json"
    ),
    dataset_manifest_path: Path = Path(
        "metadata/dataset_manifests/tess_live_search_v1.json"
    ),
    repo_root: Path = Path("."),
    replace_preparation: bool = False,
    full_sweep: bool = False,
    exclude_known_variables: bool = False,
    search_log_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch a ranked target list and scan each star in priority order.

    Already-scanned stars (from *log_path*), TOI stars, community TOI (CTOI)
    stars, and confirmed transiting planet hosts from the NASA Exoplanet Archive
    are excluded before scanning begins. Normal scans preserve the historical
    fail-open behavior for exclusion sources; metadata preparation fails closed
    so an incomplete source cannot silently enter a frozen queue. The log is
    updated after every star so progress is never lost on interruption.

    ``full_sweep``, ``exclude_known_variables``, and ``search_log_path`` are
    new, opt-in (default off) capabilities that do not change existing
    default behavior or the reproducibility of any prior frozen batch:

    - ``full_sweep=True`` passes through to :func:`select_targets`, querying
      every configured tile before ranking instead of stopping early once a
      small buffer is collected — see that function's docstring.
    - ``exclude_known_variables=True`` removes candidates already flagged as
      known variable stars in the pinned ASAS-SN Catalog X source (a live,
      zero-payload exact-TIC query) after selection. This can return fewer
      than *n_targets* results; the count removed is reported and written to
      *search_log_path* if given, never silently dropped.
    - ``search_log_path``, if given, writes a durable JSON record of exactly
      what was searched (tiles queried, coverage, raw/excluded/final counts,
      elapsed time) so a later session can see the real search extent
      instead of assuming one.

    Args:
        log_path: Path to the persistent JSON scan log.
        n_targets: Maximum stars to scan in this run.
        tmag_range: ``(min_tmag, max_tmag)`` for the TIC catalog query.
            Default ``(12.0, 14.5)`` targets the faint-star novelty frontier.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        min_snr: Minimum BLS SNR threshold.
        max_peaks: Maximum signals per star.
        max_period_grid_points: Maximum BLS trial periods per peak. Default
            20,000 keeps long-baseline QLP scans bounded.
        scorer: Scoring model.
        model_path: XGBoost model path (for xgboost/ensemble scorer).
        pipeline: MAST pipeline/author to fetch. Default QLP favors FFI-based
            discovery coverage over SPOC target-limited light curves.
        exptime: MAST exposure hint.
        query_radius_deg: Cone-search radius for each bounded TIC target tile.
        max_target_query_tiles: Maximum number of TIC sky tiles to query.
        workers: Maximum concurrent target scans or metadata inspections.
            Default 6 follows the active repo-tab worker convention.
        request_delay: Minimum seconds between worker scan starts.
        prepare_only: Freeze the metadata-only queue and manifests, then stop
            before any raw product download or pipeline execution.
    """
    log: ScanLog | None = None if prepare_only else ScanLog(log_path)

    print("Loading TESS TOI exclusion list …", flush=True)
    if prepare_only:
        toi_ids = _load_toi_tic_ids(strict=True)
    else:
        try:
            toi_ids = _load_toi_tic_ids()
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: could not load TOI list ({exc}); skipping TOI exclusion",
                flush=True,
            )
            toi_ids = set()

    print("Loading CTOI exclusion list …", flush=True)
    ctoi_ids = (
        _load_ctoi_tic_ids(strict=True) if prepare_only else _load_ctoi_tic_ids()
    )

    print("Loading confirmed transiting planet hosts …", flush=True)
    if prepare_only:
        confirmed_ids = _load_confirmed_host_tic_ids(strict=True)
    else:
        confirmed_ids = _load_confirmed_host_tic_ids()

    prior_discovery_ids = _load_prior_discovery_tic_ids(
        log_path.parent,
        strict=prepare_only,
    )
    if prepare_only:
        if log_path.exists():
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))
            already_scanned = {
                int(value) for value in log_payload.get("entries", {})
            }
        else:
            already_scanned = set()
    else:
        assert log is not None
        already_scanned = log.scanned_ids()
    exclude = (
        toi_ids
        | ctoi_ids
        | confirmed_ids
        | prior_discovery_ids
        | already_scanned
    )
    print(
        f"Excluding {len(toi_ids):,} TOI  |  {len(ctoi_ids):,} CTOI  |  "
        f"{len(confirmed_ids):,} confirmed hosts  |  "
        f"{len(prior_discovery_ids):,} historical discovery targets  |  "
        f"{len(already_scanned):,} already-scanned",
        flush=True,
    )

    print(
        f"Querying TIC for up to {n_targets} targets "
        f"(Tmag {tmag_range[0]:.1f}–{tmag_range[1]:.1f}; "
        f"radius={query_radius_deg:.2f} deg; tiles≤{max_target_query_tiles}) …",
        flush=True,
    )
    search_log: dict[str, Any] = {}
    try:
        targets = select_targets(
            n=n_targets,
            tmag_range=tmag_range,
            exclude_tic_ids=exclude,
            query_radius_deg=query_radius_deg,
            max_tiles=max_target_query_tiles,
            full_sweep=full_sweep,
            search_log=search_log,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Target selection failed: {exc}", file=sys.stderr, flush=True)
        raise

    asassn_variable_ids: frozenset[int] = frozenset()
    if exclude_known_variables and targets:
        print("Checking selected targets against known ASAS-SN variables …", flush=True)
        asassn_variable_ids = _load_asassn_variable_tic_ids(
            [t["tic_id"] for t in targets]
        )
        if asassn_variable_ids:
            before = len(targets)
            targets = [t for t in targets if t["tic_id"] not in asassn_variable_ids]
            print(
                f"  Excluded {before - len(targets)} known ASAS-SN variable(s); "
                f"{len(targets)} of the originally requested {n_targets} remain "
                "(not backfilled).",
                flush=True,
            )
    search_log["asassn_variables_excluded"] = len(asassn_variable_ids)
    search_log["final_target_count"] = len(targets)
    search_log["requested_target_count"] = n_targets

    if search_log_path is not None:
        search_log_path.parent.mkdir(parents=True, exist_ok=True)
        search_log_path.write_text(json.dumps(search_log, indent=2, default=str))
        print(f"Search manifest written to {search_log_path}", flush=True)

    if prepare_only:
        return prepare_live_search_snapshot(
            targets,
            queue_path=queue_path,
            immutable_snapshot_path=immutable_snapshot_path,
            batch_manifest_path=batch_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
            repo_root=repo_root,
            mission=mission,
            pipeline=pipeline,
            exptime=exptime,
            workers=workers,
            replace=replace_preparation,
        )
    assert log is not None
    worker_count = max(1, min(workers, len(targets)))
    print(
        f"Selected {len(targets)} candidate targets  "
        f"| pipeline={pipeline}  exptime={exptime}  "
        f"| max_peaks={max_peaks}  period_grid≤{max_period_grid_points or 'auto'}  "
        f"| workers={worker_count}  request_delay={request_delay:.2f}s\n",
        flush=True,
    )

    if not targets:
        print(
            "No candidate targets selected. Try increasing --max-target-query-tiles "
            "or --query-radius-deg after checking live-service status.",
            flush=True,
        )
        return {
            "mode": "scan",
            "items_processed": 0,
            "items_written": 0,
            "items_failed": 0,
            "output_paths": (str(log_path),),
        }

    start_time = time.monotonic()
    n_done = 0
    n_failed = 0
    limiter = _StartRateLimiter(request_delay)
    print_lock = threading.Lock()

    def scan_target(target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        limiter.wait()
        tic_id = target["tic_id"]
        log.mark_started(tic_id, target, pipeline=pipeline, exptime=exptime)
        with print_lock:
            print(
                f"[start] TIC {tic_id}  Tmag={target['tmag']:.1f}  "
                f"priority={target['priority']:.3f}  pipeline={pipeline}  "
                f"exptime={exptime}  active={log.summary()['active']}  "
                f"{_progress_suffix(n_done, len(targets), start_time)}",
                flush=True,
            )
        result = scan_star(
            tic_id,
            mission=mission,
            log=None,
            min_snr=min_snr,
            max_peaks=max_peaks,
            max_period_grid_points=max_period_grid_points,
            scorer=scorer,
            model_path=model_path,
            pipeline=pipeline,
            exptime=exptime,
            priority=target["priority"],
        )
        return target, result

    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures: list[Future[tuple[dict[str, Any], dict[str, Any]]]] = []
    try:
        futures = [executor.submit(scan_target, target) for target in targets]
        for future in as_completed(futures):
            target, result = future.result()
            tic_id = target["tic_id"]
            log.record(tic_id, result["status"], result)
            n_done += 1
            if result["status"] == "error":
                n_failed += 1
            with print_lock:
                print(
                    f"[{n_done}/{len(targets)}] TIC {tic_id}  "
                    f"Tmag={target['tmag']:.1f}  priority={target['priority']:.3f}  "
                    f"{_status_text(result)}  active={log.summary()['active']}  "
                    f"{_progress_suffix(n_done, len(targets), start_time)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        print("\nScan interrupted.", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    summary = log.summary()
    print(
        f"\nDone — {summary['total']:,} total  "
        f"| {summary['candidate_found']:,} candidates  "
        f"| {summary['scanned_clear']:,} clear  "
        f"| {summary['no_data']:,} no-data  "
        f"| {summary['error']:,} errors  "
        f"| {summary['active']:,} active",
        flush=True,
    )
    return {
        "mode": "scan",
        "items_processed": n_done,
        "items_written": n_done - n_failed,
        "items_failed": n_failed,
        "output_paths": (str(log_path),),
        "search_log": search_log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--target", type=int, metavar="TIC_ID",
        help="Scan a single star by TIC ID and exit",
    )
    mode.add_argument(
        "--summary", action="store_true",
        help="Print scan log summary and exit (no scanning)",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Write a metadata-only live-search queue and manifests, then exit "
            "before any light-curve download"
        ),
    )
    mode.add_argument(
        "--execute-prepared-batch",
        action="store_true",
        help="Scan only the frozen manifest queue and write candidate-ledger v2",
    )
    p.add_argument(
        "--log", default="logs/scan_log.json",
        help="Path to scan log JSON (default: logs/scan_log.json)",
    )
    p.add_argument(
        "--max-stars", type=int, default=500,
        help="Maximum stars to scan in background mode (default: 500)",
    )
    p.add_argument("--tmag-min", type=float, default=12.0,
                   help="Minimum TESS magnitude (default: 12.0)")
    p.add_argument("--tmag-max", type=float, default=14.5,
                   help="Maximum TESS magnitude (default: 14.5)")
    p.add_argument("--mission", default="TESS", choices=["TESS", "Kepler", "K2"])
    p.add_argument("--min-snr", type=float, default=5.0,
                   help="Minimum BLS SNR threshold (default: 5.0)")
    p.add_argument("--max-peaks", type=int, default=5,
                   help="Maximum signals per star (default: 5)")
    p.add_argument("--max-period-grid-points", type=int, default=20_000,
                   help="Maximum BLS trial periods per peak (default: 20000)")
    p.add_argument("--pipeline", default="QLP", choices=["SPOC", "QLP", "TGLC"],
                   help="MAST pipeline/author for TESS light curves (default: QLP)")
    p.add_argument("--exptime", default="long", choices=["long", "short", "fast"],
                   help="MAST exposure hint for TESS light curves (default: long)")
    p.add_argument("--query-radius-deg", type=float, default=0.5,
                   help="TIC cone-search radius per target-selection tile (default: 0.5)")
    p.add_argument("--max-target-query-tiles", type=int, default=126,
                   help="Maximum TIC sky tiles to query for target selection (default: 126, "
                        "the full configured grid)")
    p.add_argument(
        "--full-sweep", action="store_true",
        help="Query every configured tile before ranking targets, instead of "
             "stopping early once a small buffer is collected. Slower but "
             "'top N' then genuinely means top-N across the whole swept area.",
    )
    p.add_argument(
        "--exclude-known-variables", action="store_true",
        help="Remove selected targets already flagged as known variable stars "
             "in the pinned ASAS-SN Catalog X source (live, zero-payload check).",
    )
    p.add_argument(
        "--search-log-path", default=None,
        help="Write a JSON manifest of exactly what was searched (tiles, "
             "coverage, candidate/exclusion counts, elapsed time) to this path.",
    )
    p.add_argument("--workers", type=int, default=6,
                   help="Concurrent target scans/metadata queries (default: 6)")
    p.add_argument("--request-delay", type=float, default=0.5,
                   help="Minimum seconds between worker scan starts (default: 0.5)")
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Prepared-scan heartbeat interval in seconds (default: 30)",
    )
    p.add_argument(
        "--scorer", default="bayesian", choices=["bayesian", "xgboost", "ensemble"],
    )
    p.add_argument(
        "--model-path", default=None,
        help="XGBoost model JSON (required for --scorer xgboost/ensemble)",
    )
    p.add_argument(
        "--queue-path",
        default="data_selection/target_priority_queue.csv",
        help="Canonical live-search target queue CSV",
    )
    p.add_argument(
        "--queue-snapshot-path",
        default="data_selection/target_priority_queue.tess_live_search_v1.csv",
        help="Immutable queued-target snapshot CSV",
    )
    p.add_argument(
        "--batch-manifest-path",
        default="data_selection/batch_manifests/tess_live_search_v1.json",
        help="Metadata-only live-search batch manifest JSON",
    )
    p.add_argument(
        "--dataset-manifest-path",
        default="metadata/dataset_manifests/tess_live_search_v1.json",
        help="Validated live-search DatasetManifest JSON",
    )
    p.add_argument(
        "--candidate-db-path",
        type=Path,
        default=Path("data/candidates.sqlite3"),
        help="Schema-v2 candidate ledger SQLite path",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Prepared-batch process shard index (default: 0)",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Prepared-batch process shard count (default: 1)",
    )
    p.add_argument(
        "--replace-preparation",
        action="store_true",
        help="Explicitly replace preparation artifacts with the same versioned paths",
    )
    p.add_argument(
        "--no-git-report",
        action="store_true",
        help="Skip the required run-report append and commit/push step",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Run-report ledger directory",
    )
    return p.parse_args(argv)


def _write_run_report(
    *,
    started_at: str,
    elapsed_seconds: float,
    items_processed: int,
    items_written: int,
    items_failed: int,
    output_paths: tuple[str, ...],
    report_dir: Path,
    notes: str = "",
    shard_index: int | None = None,
    shard_count: int | None = None,
    git_run_fn: Any = None,
) -> None:
    """Append and safely publish one star-scanner completion report."""
    report = RunReport(
        script="star_scanner",
        status="success" if items_failed == 0 else "partial",
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed_seconds,
        items_processed=items_processed,
        items_written=items_written,
        items_failed=items_failed,
        output_paths=output_paths,
        notes=notes,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    path = report_path_for(
        "star_scanner",
        shard_index=shard_index,
        shard_count=shard_count,
        report_dir=report_dir,
    )
    kwargs: dict[str, Any] = {}
    if git_run_fn is not None:
        kwargs["run_fn"] = git_run_fn
    ok = run_and_commit_report(report, path, **kwargs)
    if ok:
        print(f"Run report committed and pushed: {path}", flush=True)
    else:
        print(
            f"Warning: run report written to {path} but commit/push failed",
            file=sys.stderr,
            flush=True,
        )


def main(argv: list[str] | None = None, *, git_run_fn: Any = None) -> int:
    """Run the scanner CLI and return a process exit code."""
    args = _parse_args(argv)
    _log_path = Path(args.log)
    _model_path = Path(args.model_path) if args.model_path else None

    if args.summary:
        if not _log_path.exists():
            print(f"No scan log found at {_log_path}", file=sys.stderr)
            return 1
        _s = ScanLog(_log_path).summary()
        print(f"Scan log: {_log_path}")
        print(f"  Total scanned     : {_s['total']:,}")
        print(f"  Candidates found  : {_s['candidate_found']:,}")
        print(f"  Clear (no signal) : {_s['scanned_clear']:,}")
        print(f"  No data           : {_s['no_data']:,}")
        print(f"  Errors            : {_s['error']:,}")
        return 0

    if args.execute_prepared_batch:
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        bundle = load_prepared_live_search_bundle(
            dataset_manifest_path=Path(args.dataset_manifest_path),
            batch_manifest_path=Path(args.batch_manifest_path),
            repo_root=_REPO_ROOT,
        )
        summary = run_prepared_live_search(
            bundle,
            log_path=_log_path,
            candidate_db_path=args.candidate_db_path,
            workers=args.workers,
            request_delay=args.request_delay,
            min_snr=args.min_snr,
            max_peaks=args.max_peaks,
            max_period_grid_points=args.max_period_grid_points,
            scorer=args.scorer,
            model_path=_model_path,
            pipeline=args.pipeline,
            exptime=args.exptime,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            heartbeat_interval_seconds=args.heartbeat_seconds,
        )
        if not args.no_git_report:
            _write_run_report(
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                items_processed=int(summary["items_processed"]),
                items_written=int(summary["items_written"]),
                items_failed=int(summary["items_failed"]),
                output_paths=tuple(summary["output_paths"]),
                report_dir=args.report_dir,
                notes=(
                    f"prepared batch {bundle.dataset_id}; shard "
                    f"{args.shard_index}/{args.shard_count}"
                ),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                git_run_fn=git_run_fn,
            )
        return 0 if int(summary["items_failed"]) == 0 else 2

    if args.target:
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        _result = run_target_scan(
            _log_path,
            args.target,
            mission=args.mission,
            min_snr=args.min_snr,
            max_peaks=args.max_peaks,
            max_period_grid_points=args.max_period_grid_points,
            scorer=args.scorer,
            model_path=_model_path,
            pipeline=args.pipeline,
            exptime=args.exptime,
        )
        if _result["status"] == "candidate_found":
            print(
                f"CANDIDATE: {_result['n_signals']} signal(s)  "
                f"best period={_result['best_period_days']:.2f} d  "
                f"FPP={_result['best_fpp']:.3f}  "
                f"pathway={_result['best_pathway']}"
            )
        elif _result["status"] == "error":
            print(f"Error: {_result['error_message']}", file=sys.stderr)
        else:
            print(f"No candidates found (status: {_result['status']})")
        if not args.no_git_report:
            is_error = _result["status"] == "error"
            _write_run_report(
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                items_processed=1,
                items_written=0 if is_error else 1,
                items_failed=1 if is_error else 0,
                output_paths=(str(_log_path),),
                report_dir=args.report_dir,
                notes="single-target scan",
                git_run_fn=git_run_fn,
            )
        if _result["status"] == "error":
            return 2
        return 0

    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    summary = run_background_scan(
        _log_path,
        n_targets=args.max_stars,
        tmag_range=(args.tmag_min, args.tmag_max),
        mission=args.mission,
        min_snr=args.min_snr,
        max_peaks=args.max_peaks,
        max_period_grid_points=args.max_period_grid_points,
        scorer=args.scorer,
        model_path=_model_path,
        pipeline=args.pipeline,
        exptime=args.exptime,
        query_radius_deg=args.query_radius_deg,
        max_target_query_tiles=args.max_target_query_tiles,
        workers=args.workers,
        request_delay=args.request_delay,
        prepare_only=args.prepare_only,
        queue_path=Path(args.queue_path),
        immutable_snapshot_path=Path(args.queue_snapshot_path),
        batch_manifest_path=Path(args.batch_manifest_path),
        dataset_manifest_path=Path(args.dataset_manifest_path),
        repo_root=_SKILLS_DIR.parent,
        replace_preparation=args.replace_preparation,
        full_sweep=args.full_sweep,
        exclude_known_variables=args.exclude_known_variables,
        search_log_path=Path(args.search_log_path) if args.search_log_path else None,
    )
    if not args.no_git_report:
        if args.prepare_only:
            output_paths = (
                summary["queue_path"],
                summary["immutable_snapshot_path"],
                summary["batch_manifest_path"],
                summary["dataset_manifest_path"],
            )
            _write_run_report(
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                items_processed=int(summary["inspected_count"]),
                items_written=int(summary["target_count"]),
                items_failed=0,
                output_paths=output_paths,
                report_dir=args.report_dir,
                notes=(
                    "metadata-only preparation; "
                    f"{summary['rejected_count']} targets rejected by policy"
                ),
                git_run_fn=git_run_fn,
            )
        else:
            search_log = summary.get("search_log") or {}
            notes = "bounded background scan"
            if search_log:
                notes += (
                    f"; tiles_queried={search_log.get('tiles_queried')}"
                    f"/{search_log.get('tiles_configured')}"
                    f" full_sweep={search_log.get('full_sweep')}"
                    f" sky_coverage_deg2={search_log.get('sky_coverage_deg2')}"
                    f" candidates_before_exclusion="
                    f"{search_log.get('raw_candidates_before_exclusion')}"
                    f" asassn_variables_excluded="
                    f"{search_log.get('asassn_variables_excluded')}"
                )
            _write_run_report(
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                items_processed=int(summary["items_processed"]),
                items_written=int(summary["items_written"]),
                items_failed=int(summary["items_failed"]),
                output_paths=tuple(summary["output_paths"]),
                report_dir=args.report_dir,
                notes=notes,
                git_run_fn=git_run_fn,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
