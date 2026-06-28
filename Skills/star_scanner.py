"""Optimal star selection and background scanning for exoplanet transit search.

Two operating modes:

  1. **Single-target** (``--target <TIC_ID>``): analyse one star, log result.
  2. **Background scan** (default): query the TESS Input Catalog for
     high-priority uncharacterised stars, rank them, and scan in order until
     stopped (Ctrl-C) or ``--max-stars`` is reached.

Stars already in the TESS TOI disposition list (known objects being actively
followed up) and stars already present in the scan log are skipped
automatically.  Scanning can be interrupted and resumed at any time — the log
records which TIC IDs have been processed.

Usage
-----
    # Single target
    python Skills/star_scanner.py --target 150428135

    # Background scan (auto-resumes via log)
    python Skills/star_scanner.py --log data/scan_log.json --max-stars 1000

    # Show scan log summary without scanning
    python Skills/star_scanner.py --summary --log data/scan_log.json

    # Narrow magnitude window; use ML scorer
    python Skills/star_scanner.py --log data/scan_log.json \\
        --tmag-min 11 --tmag-max 13 \\
        --scorer xgboost --model-path data/model.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.cli import run_pipeline

# ExoFOP TOI table (same endpoint used by fetch_tess_toi.py)
_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

_SKILLS_DIR = Path(__file__).resolve().parent

_DEFAULT_SEARCH_CENTERS: tuple[tuple[float, float], ...] = tuple(
    (float(ra), float(dec))
    for dec in (-60, -30, 0, 30, 60)
    for ra in range(0, 360, 30)
)


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

# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def priority_score(
    tmag: float,
    teff: float | None = None,
    n_sectors: int | None = None,
    contratio: float | None = None,
) -> float:
    """Compute a [0, 1] priority score for a TIC star.

    Higher score = more promising target for planet transit search.

    Weighted sub-scores:

    * **Magnitude** (0.30): peaks at Tmag ≈ 12–13; falls off outside [10, 14].
    * **Stellar type** (0.25): prefers K/M dwarfs (3000–5500 K) where the
      habitable zone lies at short, easily-observed periods.
    * **Sector coverage** (0.25): more sectors → more transits visible; capped
      at 6 sectors for score = 1.0.  ``None`` → neutral 0.5.
    * **Contamination** (0.20): lower contamination → cleaner transit depth;
      ``contratio = 0`` scores 1.0.  ``None`` → neutral 0.5.

    Args:
        tmag: TESS magnitude.
        teff: Effective temperature in Kelvin (``None`` → neutral 0.5).
        n_sectors: Sectors of TESS data available (``None`` → neutral 0.5).
        contratio: Fraction of aperture flux from nearby sources
            (``None`` → neutral 0.5).

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

    return (
        0.30 * mag_score
        + 0.25 * teff_score
        + 0.25 * sector_score
        + 0.20 * cont_score
    )


# ---------------------------------------------------------------------------
# Persistent scan log
# ---------------------------------------------------------------------------


class ScanLog:
    """Read/write a JSON scan log tracking which stars have been analysed.

    Log schema::

        {
          "last_updated": "<ISO-8601 UTC>",
          "entries": {
            "<tic_id>": {
              "tic_id": int,
              "scanned_at": "<ISO-8601 UTC>",
              "status": "candidate_found|scanned_clear|no_data|error",
              "n_signals": int,
              "best_period_days": float | null,
              "best_fpp": float | null,
              "best_pathway": str | null,
              "priority_score": float | null,
              "error_message": str | null
            }
          }
        }

    Writes are atomic: the file is written to a ``.tmp`` sibling and then
    renamed, so a partial write never corrupts the log.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, Any] = {"last_updated": "", "entries": {}}
        if path.exists():
            with path.open() as fh:
                self._data = json.load(fh)

    def is_scanned(self, tic_id: int) -> bool:
        """Return True if *tic_id* has an entry in the log."""
        return str(tic_id) in self._data["entries"]

    def scanned_ids(self) -> set[int]:
        """Return the set of all TIC IDs recorded in the log."""
        return {int(k) for k in self._data["entries"]}

    def record(self, tic_id: int, status: str, result: dict[str, Any]) -> None:
        """Add or overwrite the log entry for *tic_id*."""
        entry: dict[str, Any] = {
            "tic_id": tic_id,
            "scanned_at": datetime.now(UTC).isoformat(),
            "status": status,
            "n_signals": result.get("n_signals", 0),
            "best_period_days": result.get("best_period_days"),
            "best_fpp": result.get("best_fpp"),
            "best_pathway": result.get("best_pathway"),
            "priority_score": result.get("priority_score"),
            "error_message": result.get("error_message"),
        }
        self._data["entries"][str(tic_id)] = entry
        self._data["last_updated"] = datetime.now(UTC).isoformat()
        self._flush()

    def summary(self) -> dict[str, int]:
        """Return a dict of status → count plus a ``"total"`` key."""
        counts: dict[str, int] = {
            "candidate_found": 0,
            "scanned_clear": 0,
            "no_data": 0,
            "error": 0,
            "total": 0,
        }
        for entry in self._data["entries"].values():
            status = entry.get("status", "error")
            if status in counts:
                counts[status] += 1
            counts["total"] += 1
        return counts

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump(self._data, fh, indent=2)
        tmp.replace(self._path)


# ---------------------------------------------------------------------------
# Exclusion lists: TOI, CTOI, confirmed planets
# ---------------------------------------------------------------------------


def _load_toi_tic_ids() -> set[int]:
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
        return set()
    return {int(v) for v in df[tic_col].dropna()}


def _load_ctoi_tic_ids() -> set[int]:
    """Download the ExoFOP CTOI table and return the set of TIC IDs it contains.

    Uses ``Skills/fetch_exofop_ctoi.py`` with its injectable fetch function.
    Returns an empty set on any failure so the scan can continue.
    """
    import importlib.util

    ctoi_path = _SKILLS_DIR / "fetch_exofop_ctoi.py"
    spec = importlib.util.spec_from_file_location("fetch_exofop_ctoi", ctoi_path)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        result = module.fetch_ctoi_table()
        return {int(r["tic_id"]) for r in result.rows if r.get("tic_id")}
    except Exception:  # noqa: BLE001
        return set()


def _load_confirmed_host_tic_ids() -> frozenset[int]:
    """Return TIC IDs of confirmed transiting planet hosts from NASA Exoplanet Archive.

    Uses ``Skills/fetch_confirmed_hosts.py``.  Returns an empty frozenset on failure.
    """
    import importlib.util

    host_path = _SKILLS_DIR / "fetch_confirmed_hosts.py"
    spec = importlib.util.spec_from_file_location("fetch_confirmed_hosts", host_path)
    if spec is None or spec.loader is None:
        return frozenset()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        return module.fetch_confirmed_host_tic_ids()
    except Exception:  # noqa: BLE001
        return frozenset()


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def select_targets(
    n: int = 100,
    tmag_range: tuple[float, float] = (12.0, 14.5),
    exclude_tic_ids: set[int] | None = None,
    *,
    query_radius_deg: float = 0.5,
    max_tiles: int = 60,
    retry_attempts: int = 2,
    retry_delay: float = 2.0,
) -> list[dict[str, Any]]:
    """Query the TIC catalog and return up to *n* stars ranked by priority.

    Stars in *exclude_tic_ids* (e.g. known TOIs or already-scanned stars) are
    removed before ranking.  The query is intentionally tiled by sky position:
    all-sky TIC magnitude queries are too large for MAST and can be closed by
    the remote service before a response is returned.

    Args:
        n: Maximum number of targets to return.
        tmag_range: ``(min_tmag, max_tmag)`` magnitude range for the query.
        exclude_tic_ids: TIC IDs to omit from the results.
        query_radius_deg: Cone-search radius for each TIC tile.
        max_tiles: Maximum number of sky tiles to query.
        retry_attempts: Attempts per tile before skipping it.
        retry_delay: Seconds to wait between retry attempts.

    Returns:
        List of dicts, sorted by ``"priority"`` descending, each with keys:
        ``tic_id``, ``tmag``, ``teff``, ``contratio``, ``priority``.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    exclude = exclude_tic_ids or set()
    min_tmag, max_tmag = tmag_range
    target_pool_size = max(n * 3, n)

    targets: list[dict[str, Any]] = []
    seen: set[int] = set()
    tile_errors: list[str] = []

    for tile_idx, (ra_deg, dec_deg) in enumerate(_DEFAULT_SEARCH_CENTERS[:max_tiles], 1):
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        result: Any | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                result = Catalogs.query_region(
                    coord,
                    radius=query_radius_deg * u.deg,
                    catalog="TIC",
                )
                break
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"tile {tile_idx} ra={ra_deg:.1f} dec={dec_deg:.1f} "
                    f"attempt {attempt}/{retry_attempts}: {exc}"
                )
                tile_errors.append(msg)
                if attempt < retry_attempts:
                    time.sleep(retry_delay)

        if result is None:
            continue

        for row in result:
            obj_type = _row_get(row, "objType")
            if obj_type is not None and str(obj_type).strip().upper() != "STAR":
                continue
            try:
                tic_id = int(_row_get(row, "ID"))
            except (ValueError, TypeError):
                continue
            if tic_id in exclude or tic_id in seen:
                continue

            try:
                tmag = float(_row_get(row, "Tmag"))
            except (ValueError, TypeError):
                continue
            if tmag != tmag or not (min_tmag <= tmag <= max_tmag):  # NaN guard
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

            pri = priority_score(tmag, teff=teff, contratio=contratio)
            seen.add(tic_id)
            targets.append(
                {
                    "tic_id": tic_id,
                    "tmag": tmag,
                    "teff": teff,
                    "contratio": contratio,
                    "priority": pri,
                }
            )
        if len(targets) >= target_pool_size:
            break

    if not targets and tile_errors:
        sample = "; ".join(tile_errors[-3:])
        raise RuntimeError(
            f"TIC target selection failed across {len(tile_errors)} attempts: {sample}"
        )

    targets.sort(key=lambda t: t["priority"], reverse=True)
    return targets[:n]


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
    scorer: str = "bayesian",
    model_path: Path | None = None,
    priority: float | None = None,
) -> dict[str, Any]:
    """Run the full pipeline on one star and optionally persist the result.

    Args:
        tic_id: Numeric TIC identifier.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        log: Optional :class:`ScanLog`; when provided the result is recorded.
        min_snr: Minimum BLS SNR threshold.
        max_peaks: Maximum signals to search for per star.
        scorer: ``"bayesian"``, ``"xgboost"``, or ``"ensemble"``.
        model_path: XGBoost model JSON (required for xgboost/ensemble).
        priority: Pre-computed priority score to store in the log entry.

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
        "priority_score": priority,
        "error_message": None,
    }

    try:
        rows = run_pipeline(
            target_id,
            mission,  # type: ignore[arg-type]
            min_snr=min_snr,
            max_peaks=max_peaks,
            scorer=scorer,
            model_path=model_path,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        result["status"] = "no_data" if _is_no_data_error(message) else "error"
        result["error_message"] = message
        if log is not None:
            log.record(tic_id, result["status"], result)
        return result

    if not rows:
        result["status"] = "scanned_clear"
    else:
        result["status"] = "candidate_found"
        result["n_signals"] = len(rows)
        best = min(rows, key=lambda r: r["scores"]["false_positive_probability"])
        result["best_period_days"] = best["period_days"]
        result["best_fpp"] = best["scores"]["false_positive_probability"]
        result["best_pathway"] = best["pathway"]

    if log is not None:
        log.record(tic_id, result["status"], result)
    return result


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
    scorer: str = "bayesian",
    model_path: Path | None = None,
    query_radius_deg: float = 0.5,
    max_target_query_tiles: int = 60,
) -> None:
    """Fetch a ranked target list and scan each star in priority order.

    Already-scanned stars (from *log_path*), TOI stars, community TOI (CTOI)
    stars, and confirmed transiting planet hosts from the NASA Exoplanet Archive
    are excluded before scanning begins.  All exclusion lists fail open — if a
    list cannot be fetched the scan continues without that exclusion.  The log
    is updated after every star so progress is never lost on interruption.

    Args:
        log_path: Path to the persistent JSON scan log.
        n_targets: Maximum stars to scan in this run.
        tmag_range: ``(min_tmag, max_tmag)`` for the TIC catalog query.
            Default ``(12.0, 14.5)`` targets the faint-star novelty frontier.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        min_snr: Minimum BLS SNR threshold.
        max_peaks: Maximum signals per star.
        scorer: Scoring model.
        model_path: XGBoost model path (for xgboost/ensemble scorer).
        query_radius_deg: Cone-search radius for each bounded TIC target tile.
        max_target_query_tiles: Maximum number of TIC sky tiles to query.
    """
    log = ScanLog(log_path)

    print("Loading TESS TOI exclusion list …", flush=True)
    try:
        toi_ids = _load_toi_tic_ids()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load TOI list ({exc}); skipping TOI exclusion", flush=True)
        toi_ids = set()

    print("Loading CTOI exclusion list …", flush=True)
    try:
        ctoi_ids = _load_ctoi_tic_ids()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load CTOI list ({exc}); skipping CTOI exclusion", flush=True)
        ctoi_ids = set()

    print("Loading confirmed transiting planet hosts …", flush=True)
    try:
        confirmed_ids = _load_confirmed_host_tic_ids()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load confirmed hosts ({exc}); skipping", flush=True)
        confirmed_ids = frozenset()

    already_scanned = log.scanned_ids()
    exclude = toi_ids | ctoi_ids | confirmed_ids | already_scanned
    print(
        f"Excluding {len(toi_ids):,} TOI  |  {len(ctoi_ids):,} CTOI  |  "
        f"{len(confirmed_ids):,} confirmed hosts  |  "
        f"{len(already_scanned):,} already-scanned",
        flush=True,
    )

    print(
        f"Querying TIC for up to {n_targets} targets "
        f"(Tmag {tmag_range[0]:.1f}–{tmag_range[1]:.1f}; "
        f"radius={query_radius_deg:.2f} deg; tiles≤{max_target_query_tiles}) …",
        flush=True,
    )
    try:
        targets = select_targets(
            n=n_targets,
            tmag_range=tmag_range,
            exclude_tic_ids=exclude,
            query_radius_deg=query_radius_deg,
            max_tiles=max_target_query_tiles,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Target selection failed: {exc}", file=sys.stderr, flush=True)
        raise
    print(f"Selected {len(targets)} candidate targets\n", flush=True)

    if not targets:
        print(
            "No candidate targets selected. Try increasing --max-target-query-tiles "
            "or --query-radius-deg after checking live-service status.",
            flush=True,
        )
        return

    try:
        for idx, target in enumerate(targets, 1):
            tic_id = target["tic_id"]
            pri = target["priority"]
            print(
                f"[{idx}/{len(targets)}] TIC {tic_id}  "
                f"Tmag={target['tmag']:.1f}  priority={pri:.3f} …",
                end=" ",
                flush=True,
            )
            result = scan_star(
                tic_id,
                mission=mission,
                log=log,
                min_snr=min_snr,
                max_peaks=max_peaks,
                scorer=scorer,
                model_path=model_path,
                priority=pri,
            )
            status = result["status"]
            if status == "candidate_found":
                print(
                    f"CANDIDATE  P={result['best_period_days']:.2f} d  "
                    f"FPP={result['best_fpp']:.3f}  [{result['best_pathway']}]",
                    flush=True,
                )
            elif status == "scanned_clear":
                print("clear", flush=True)
            elif status == "no_data":
                print("no data", flush=True)
            else:
                print(f"error: {result['error_message']}", flush=True)
    except KeyboardInterrupt:
        print("\nScan interrupted.", flush=True)

    summary = log.summary()
    print(
        f"\nDone — {summary['total']:,} total  "
        f"| {summary['candidate_found']:,} candidates  "
        f"| {summary['scanned_clear']:,} clear  "
        f"| {summary['no_data']:,} no-data  "
        f"| {summary['error']:,} errors",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
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
    p.add_argument(
        "--log", default="data/scan_log.json",
        help="Path to scan log JSON (default: data/scan_log.json)",
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
    p.add_argument("--query-radius-deg", type=float, default=0.5,
                   help="TIC cone-search radius per target-selection tile (default: 0.5)")
    p.add_argument("--max-target-query-tiles", type=int, default=60,
                   help="Maximum TIC sky tiles to query for target selection (default: 60)")
    p.add_argument(
        "--scorer", default="bayesian", choices=["bayesian", "xgboost", "ensemble"],
    )
    p.add_argument(
        "--model-path", default=None,
        help="XGBoost model JSON (required for --scorer xgboost/ensemble)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _log_path = Path(args.log)
    _model_path = Path(args.model_path) if args.model_path else None

    if args.summary:
        if not _log_path.exists():
            print(f"No scan log found at {_log_path}", file=sys.stderr)
            sys.exit(1)
        _s = ScanLog(_log_path).summary()
        print(f"Scan log: {_log_path}")
        print(f"  Total scanned     : {_s['total']:,}")
        print(f"  Candidates found  : {_s['candidate_found']:,}")
        print(f"  Clear (no signal) : {_s['scanned_clear']:,}")
        print(f"  No data           : {_s['no_data']:,}")
        print(f"  Errors            : {_s['error']:,}")
        sys.exit(0)

    if args.target:
        _log = ScanLog(_log_path)
        _result = scan_star(
            args.target,
            mission=args.mission,
            log=_log,
            min_snr=args.min_snr,
            max_peaks=args.max_peaks,
            scorer=args.scorer,
            model_path=_model_path,
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
            sys.exit(2)
        else:
            print(f"No candidates found (status: {_result['status']})")
        sys.exit(0)

    run_background_scan(
        _log_path,
        n_targets=args.max_stars,
        tmag_range=(args.tmag_min, args.tmag_max),
        mission=args.mission,
        min_snr=args.min_snr,
        max_peaks=args.max_peaks,
        scorer=args.scorer,
        model_path=_model_path,
        query_radius_deg=args.query_radius_deg,
        max_target_query_tiles=args.max_target_query_tiles,
    )
