"""Fetch TESS light curves for K2 EPIC stars to build the K2-TESS overlap corpus.

Downloads phase-folded TESS snippets for all K2 confirmed planets and confirmed
false positives that MAST has also observed with TESS.  Intended for expanding
the TESS training corpus when Kepler and TESS-only corpora are insufficient to
push CNN AUC past the 0.85 production gate.

The K2 planet catalog (k2pandc) ephemerides (period and epoch) are used to
phase-fold the TESS light curves.  K2 epochs are in BKJD (BJD − 2454833.0),
the same convention as Kepler.  TESS light curve timestamps (BTJD = BJD − 2457000)
are converted to full BJD before folding.

Output format matches ``data/tess_snippets_v2.jsonl``::

    {"tic_id": <epic_id>, "label": 0|1, "flux": [...201 floats...],
     "source": "k2_tess_overlap", "period_days": 3.4,
     "epoch_bjd": 2454900.0, "n_bins": 201, "epic_id": <epic_id>}

The ``tic_id`` field stores the K2 EPIC ID for corpus compatibility; it does
**not** represent a TESS TIC ID.  Downstream tools (``build_cnn_training_data.py``)
only use ``flux`` and ``label``, so this is safe.  The ``epic_id`` field is
present for provenance only.

Resume is automatic: rows whose ``epic_id`` + ``period_days`` key already appears
in the output JSONL are skipped.  Terminal failures are recorded in
``<output>.failures.jsonl`` and skipped on ordinary reruns; pass
``--retry-failures`` for an intentional recheck.

Run command (Mac only — requires .venv with lightkurve):
    caffeinate -dims .venv/bin/python Skills/fetch_tess_k2_overlap_snippets.py \\
        --output data/tess_k2_overlap_snippets.jsonl \\
        --workers 4 \\
        --request-delay 0.25

After download, merge with the existing TESS corpus and rebuild splits:
    cat data/tess_snippets_v2.jsonl data/tess_kepler_overlap_snippets.jsonl \\
        data/tess_k2_overlap_snippets.jsonl > data/tess_c20_combined_snippets.jsonl
    .venv/bin/python Skills/build_cnn_training_data.py \\
        data/tess_c20_combined_snippets.jsonl \\
        --output-dir data/tess_c20_cnn_splits --seed 7

Public API
----------
K2Row(epic_id, disposition, period_days, epoch_bkjd)
K2SnippetResult(epic_id, label, flux, period_days, epoch_bjd, n_bins, flag)
fetch_k2_table(url) -> list[K2Row]
build_k2_tess_snippet(row, *, n_bins, lc_fetcher) -> K2SnippetResult
build_k2_tess_snippets(rows, *, n_bins, output_path, lc_fetcher, max_errors,
                       workers, request_delay) -> int
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import socket
import ssl
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

# Prevent indefinite hangs when WiFi drops mid-download.
socket.setdefaulttimeout(120)

_K2_BJD_OFFSET = 2454833.0   # K2 epoch (BKJD) = BJD − 2454833
_TESS_BJD_OFFSET = 2457000.0  # TESS BTJD = BJD − 2457000

_K2_TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Column name candidates for k2pandc, checked in priority order against
# the live TAP schema.  NASA occasionally renames columns across archive
# versions so we discover the real names at startup rather than hardcoding.
_K2_COL_CANDIDATES: dict[str, list[str]] = {
    "epic_id":     ["epic_id"],
    "disposition": ["k2_disposition", "disp_pou", "disposition"],
    "period":      ["pl_orbper", "period"],
    "epoch":       ["pl_tranmid", "pl_tranmidj", "t0"],
}
_TERMINAL_FAILURE_FLAGS = {"NO_LIGHTKURVE", "NO_DATA", "SHORT", "NONFINITE"}


def _emit_progress(message: str) -> None:
    """Print progress without letting a damaged stdout kill a long run."""
    seen: set[int] = set()
    for stream in (sys.stdout, sys.__stdout__, sys.stderr, sys.__stderr__):
        if stream is None or id(stream) in seen:
            continue
        seen.add(id(stream))
        if getattr(stream, "closed", False):
            continue
        try:
            print(message, file=stream, flush=True)
            return
        except (OSError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class K2Row:
    """A single record from the NASA Exoplanet Archive K2 planets-and-candidates table."""

    epic_id: int
    disposition: str    # "CONFIRMED" or "FALSE POSITIVE"
    period_days: float
    epoch_bkjd: float   # BJD − 2454833


@dataclass(frozen=True)
class K2SnippetResult:
    """Outcome of processing a single K2 planet against TESS data."""

    epic_id: int
    label: int             # 1 = CONFIRMED, 0 = FALSE POSITIVE
    flux: tuple[float, ...]
    period_days: float
    epoch_bjd: float
    n_bins: int
    flag: str  # "OK" | "NO_LIGHTKURVE" | "NO_DATA" | "SHORT" | "NONFINITE" | "ERROR:..."


# ---------------------------------------------------------------------------
# Phase-fold / normalise helpers (identical to fetch_tess_kepler_overlap_snippets.py)
# ---------------------------------------------------------------------------


def _phase_fold_bin(
    time_bjd: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> list[float]:
    bin_flux: list[list[float]] = [[] for _ in range(n_bins)]
    for t, f in zip(time_bjd, flux, strict=False):
        if not math.isfinite(t) or not math.isfinite(f):
            continue
        ph = ((t - epoch) % period) / period
        ph = ph - 1.0 if ph >= 0.5 else ph
        b = int((ph + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bin_flux[b].append(f)
    return [sum(vals) / len(vals) if vals else 1.0 for vals in bin_flux]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mad(values: list[float], med: float) -> float:
    return _median([abs(v - med) for v in values])


def _normalise(flux_bins: list[float]) -> list[float]:
    if any(not math.isfinite(v) for v in flux_bins):
        return []
    med = _median(flux_bins)
    scale = _mad(flux_bins, med) * 1.4826
    if scale < 1e-10:
        return [0.0] * len(flux_bins)
    return [(v - med) / scale for v in flux_bins]


# ---------------------------------------------------------------------------
# K2 table fetcher
# ---------------------------------------------------------------------------


def _discover_k2_columns(ctx: ssl.SSLContext | None) -> dict[str, str]:
    """Query TAP schema to find the actual column names in k2pandc.

    Returns a mapping ``{canonical_name: actual_column_name}`` for the four
    fields we need.  Falls back to the first candidate for each field when the
    schema endpoint is unreachable (e.g., in offline tests).
    """
    schema_url = (
        _K2_TAP_BASE
        + "?query=select+column_name+from+tap_schema.columns"
        + "+where+table_name='k2pandc'"
        + "&format=json"
    )
    available: set[str] = set()
    try:
        with urlopen(schema_url, timeout=60, context=ctx) as resp:  # noqa: S310
            available = {
                str(row.get("column_name", "")).lower()
                for row in json.loads(resp.read())
            }
    except Exception:
        pass  # discovery failed — first candidate used as fallback below

    result: dict[str, str] = {}
    for canonical, candidates in _K2_COL_CANDIDATES.items():
        for c in candidates:
            if not available or c in available:
                result[canonical] = c
                break
        if canonical not in result:
            result[canonical] = candidates[0]
    return result


def fetch_k2_table(url: str | None = None) -> list[K2Row]:
    """Fetch the K2 planets-and-candidates table from the NASA Exoplanet Archive TAP service.

    Performs schema discovery first to find the correct column names in k2pandc,
    then executes the real query.  This makes the fetcher resilient to NASA
    archive column renames between versions.

    Args:
        url: Full TAP URL override (for tests).  When None, the URL is built
             dynamically from schema-discovered column names.

    Returns:
        List of :class:`K2Row` objects with valid period and epoch.

    Raises:
        RuntimeError: If the table cannot be fetched or parsed.
    """
    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None

    if url is not None:
        # Test override: caller provides a pre-built URL and controls column names.
        col = {"epic_id": "epic_id", "disposition": "disposition",
               "period": "period", "epoch": "t0"}
        query_url = url
    else:
        col = _discover_k2_columns(ctx)
        epic_col = col["epic_id"]
        disp_col = col["disposition"]
        period_col = col["period"]
        epoch_col = col["epoch"]
        # Fetch all rows with valid period and epoch; filter by disposition
        # locally in Python.  Omitting the IN clause from the SQL avoids
        # server-side quoting ambiguity (+ vs %20 inside SQL string literals)
        # and protects against archive disposition value renames.
        sql = (
            f"select {epic_col},{disp_col},{period_col},{epoch_col}"
            f" from k2pandc"
            f" where {period_col} is not null"
            f" and {epoch_col} is not null"
        )
        # Use urlencode (spaces → +) and csv format — NASA TAP always supports
        # csv; json with %20-encoded spaces has proven unreliable on k2pandc.
        params = urlencode({"query": sql, "format": "csv", "MAXREC": "100000"})
        query_url = f"{_K2_TAP_BASE}?{params}"
        _emit_progress(f"K2 TAP columns discovered: {col}")
        _emit_progress(f"K2 TAP query URL: {query_url}")

    try:
        with urlopen(query_url, timeout=120, context=ctx) as resp:  # noqa: S310
            raw_csv = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"K2 TAP HTTP {exc.code} ({exc.reason}): {err_body}"
        ) from exc

    reader = csv.DictReader(io.StringIO(raw_csv))
    rows: list[K2Row] = []
    for rec in reader:
        try:
            epic_id = int(rec[col["epic_id"]])
            disposition = str(rec[col["disposition"]]).strip().upper()
            period = float(rec[col["period"]])
            epoch_bkjd = float(rec[col["epoch"]])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(period) or period <= 0:
            continue
        if not math.isfinite(epoch_bkjd):
            continue
        if disposition not in {"CONFIRMED", "FALSE POSITIVE"}:
            continue
        rows.append(K2Row(
            epic_id=epic_id,
            disposition=disposition,
            period_days=period,
            epoch_bkjd=epoch_bkjd,
        ))
    return rows


# ---------------------------------------------------------------------------
# TESS light curve fetcher for a K2 EPIC star
# ---------------------------------------------------------------------------


def _default_lc_fetcher(
    epic_id: int,
    period: float,
    epoch_bjd: float,
) -> tuple[list[float], list[float]] | None:
    """Fetch a TESS light curve for a K2 star (EPIC ID) via lightkurve.

    Searches MAST for TESS SPOC or QLP data for "EPIC {epic_id}".  MAST
    resolves EPIC identifiers to sky coordinates and returns any TESS sectors
    that covered the same position.

    Returns:
        (time_bjd, flux) in full BJD, or None if no TESS data found.
    """
    try:
        import lightkurve as lk
    except ImportError:
        return None

    for author in ("SPOC", "QLP"):
        result = lk.search_lightcurve(f"EPIC {epic_id}", mission="TESS", author=author)
        if len(result) == 0:
            continue
        light_curves = []
        for idx in range(len(result.table)):
            # Avoid SearchResult.download_all(): Lightkurve decorates it with
            # suppress_stdout, which mutates process-global sys.stdout and is
            # unsafe while this script fetches EPIC groups concurrently.
            light_curves.append(
                result._download_one(  # noqa: SLF001
                    table=result.table[idx : idx + 1],
                    quality_bitmask="default",
                    download_dir=None,
                    cutout_size=None,
                )
            )
        if not light_curves:
            continue
        lc_coll = lk.LightCurveCollection(light_curves)
        lc = lc_coll.stitch()
        with contextlib.suppress(Exception):
            lc = lc.normalize()
        # TESS time is BTJD = BJD - 2457000; convert to full BJD.
        time_bjd = [float(t) + _TESS_BJD_OFFSET for t in lc.time.value]
        flux = [float(f) for f in lc.flux.value]
        return time_bjd, flux

    return None


# ---------------------------------------------------------------------------
# Per-K2-planet snippet builder
# ---------------------------------------------------------------------------


def build_k2_tess_snippet(
    row: K2Row,
    *,
    n_bins: int = 201,
    lc_fetcher: Callable | None = None,
) -> K2SnippetResult:
    """Build a single phase-folded TESS snippet for a K2 planet using its K2 ephemeris.

    Args:
        row: K2 record with period and epoch in BKJD.
        n_bins: Number of phase bins for the output snippet.
        lc_fetcher: Injectable fetcher returning (time_bjd, flux) or None.

    Returns:
        :class:`K2SnippetResult` with flag "OK" on success.
    """
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _K2_BJD_OFFSET

    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        raw = fetcher(row.epic_id, row.period_days, epoch_bjd)
    except Exception as exc:
        return K2SnippetResult(
            epic_id=row.epic_id, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag=f"ERROR:{exc}",
        )

    if raw is None:
        return _missing_light_curve_result(row, n_bins=n_bins)

    return _build_k2_tess_snippet_from_raw(row, raw, n_bins=n_bins)


def _missing_light_curve_result(row: K2Row, *, n_bins: int) -> K2SnippetResult:
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _K2_BJD_OFFSET
    try:
        import lightkurve  # noqa: F401
    except ImportError:
        return K2SnippetResult(
            epic_id=row.epic_id, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="NO_LIGHTKURVE",
        )
    return K2SnippetResult(
        epic_id=row.epic_id, label=label,
        flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
        n_bins=n_bins, flag="NO_DATA",
    )


def _build_k2_tess_snippet_from_raw(
    row: K2Row,
    raw: tuple[list[float], list[float]],
    *,
    n_bins: int,
) -> K2SnippetResult:
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _K2_BJD_OFFSET

    time_bjd, flux = raw
    finite_pairs = [
        (t, f)
        for t, f in zip(time_bjd, flux, strict=False)
        if math.isfinite(t) and math.isfinite(f)
    ]
    if len(finite_pairs) < n_bins:
        return K2SnippetResult(
            epic_id=row.epic_id, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="SHORT",
        )

    t_bjd = [p[0] for p in finite_pairs]
    f_vals = [p[1] for p in finite_pairs]
    bins = _phase_fold_bin(t_bjd, f_vals, row.period_days, epoch_bjd, n_bins)
    normalised = _normalise(bins)
    if len(normalised) != n_bins or any(not math.isfinite(v) for v in normalised):
        return K2SnippetResult(
            epic_id=row.epic_id, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="NONFINITE",
        )

    return K2SnippetResult(
        epic_id=row.epic_id, label=label,
        flux=tuple(normalised), period_days=row.period_days,
        epoch_bjd=epoch_bjd, n_bins=n_bins, flag="OK",
    )


def _process_epic_group(
    group_rows: list[K2Row],
    *,
    n_bins: int,
    lc_fetcher: Callable | None,
) -> list[K2SnippetResult]:
    """Fetch one TESS light curve for an EPIC star and fold each K2 planet locally."""
    first = group_rows[0]
    first_epoch_bjd = first.epoch_bkjd + _K2_BJD_OFFSET
    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        raw = fetcher(first.epic_id, first.period_days, first_epoch_bjd)
    except Exception as exc:
        return [
            K2SnippetResult(
                epic_id=row.epic_id,
                label=1 if row.disposition == "CONFIRMED" else 0,
                flux=(),
                period_days=row.period_days,
                epoch_bjd=row.epoch_bkjd + _K2_BJD_OFFSET,
                n_bins=n_bins,
                flag=f"ERROR:{exc}",
            )
            for row in group_rows
        ]
    if raw is None:
        return [_missing_light_curve_result(row, n_bins=n_bins) for row in group_rows]
    return [
        _build_k2_tess_snippet_from_raw(row, raw, n_bins=n_bins)
        for row in group_rows
    ]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def build_k2_tess_snippets(
    rows: list[K2Row],
    *,
    n_bins: int = 201,
    output_path: Path,
    lc_fetcher: Callable | None = None,
    max_errors: int = 50,
    failure_log_path: Path | None = None,
    retry_failures: bool = False,
    workers: int = 4,
    request_delay: float = 0.25,
) -> int:
    """Build phase-folded TESS snippets for a list of K2 rows.

    Writes to ``output_path`` in append mode (``"a"``); existing content is
    read at startup to populate the ``already_done`` resume set.  The file is
    created if it does not exist.

    Args:
        rows: K2 records to process.
        n_bins: Number of phase bins per snippet.
        output_path: Path to the JSONL output file.
        lc_fetcher: Injectable fetcher for testing.
        max_errors: Stop early after this many consecutive non-OK results.
        failure_log_path: Terminal failure JSONL sidecar path.
        retry_failures: Reprocess rows recorded in the failure sidecar.
        workers: Number of EPIC groups to process concurrently.
        request_delay: Seconds to wait between submitting EPIC fetch jobs.

    Returns:
        Number of snippets written.
    """
    workers = max(1, int(workers))
    request_delay = max(0.0, float(request_delay))
    output_path = Path(output_path)
    failure_log_path = (
        Path(failure_log_path)
        if failure_log_path is not None
        else output_path.with_name(output_path.name + ".failures.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume key: "<epic_id>:<period_days>" — handles multi-planet EPIC stars.
    def _resume_key(epic_id: int, period_days: float) -> str:
        return f"{epic_id}:{period_days:.8f}"

    # Build resume set from existing output and durable terminal failures.
    already_done: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    eid = obj.get("epic_id")
                    pd = obj.get("period_days")
                    if eid is not None and pd is not None:
                        already_done.add(_resume_key(int(eid), float(pd)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    already_failed: set[str] = set()
    if not retry_failures and failure_log_path.exists():
        with failure_log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    eid = obj.get("epic_id")
                    pd = obj.get("period_days")
                    flag = obj.get("flag")
                    if eid is not None and pd is not None and flag in _TERMINAL_FAILURE_FLAGS:
                        already_failed.add(_resume_key(int(eid), float(pd)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    pending_rows = [
        r for r in rows
        if _resume_key(r.epic_id, r.period_days) not in already_done
        and _resume_key(r.epic_id, r.period_days) not in already_failed
    ]
    pending_by_epic: dict[int, list[K2Row]] = defaultdict(list)
    for row in pending_rows:
        pending_by_epic[row.epic_id].append(row)
    pending_groups = list(pending_by_epic.values())
    n_total = len(pending_rows)
    n_groups = len(pending_groups)
    n_processed = 0
    n_written = 0
    n_skipped = 0
    n_terminal_failures = 0
    n_errors = 0
    start = time.monotonic()

    _emit_progress(
        f"K2-TESS overlap fetch: {len(rows)} total K2 planets, "
        f"{len(already_done)} already done, "
        f"{len(already_failed)} terminal failures skipped, "
        f"{n_total} pending across {n_groups} EPIC groups."
    )
    _emit_progress(
        f"Output: {output_path}  failures={failure_log_path}  "
        f"n_bins={n_bins}  max_errors={max_errors}  "
        f"workers={workers}  request_delay={request_delay:.2f}s  "
        f"retry_failures={retry_failures}"
    )

    def write_result(result: K2SnippetResult, group_index: int, group_size: int) -> None:
        nonlocal n_processed, n_written, n_skipped, n_terminal_failures, n_errors
        n_processed += 1
        elapsed = time.monotonic() - start
        rate = n_processed / elapsed if elapsed > 0 else float("inf")
        remaining = (n_total - n_processed) / rate if rate > 0 else float("inf")
        eta = (
            f"{remaining/60:.0f}m{remaining%60:.0f}s"
            if remaining > 90
            else f"{remaining:.0f}s"
        )
        group_progress = f"group={group_index}/{n_groups} group_rows={group_size}"
        if result.flag == "OK":
            record = {
                "tic_id": result.epic_id,
                "label": result.label,
                "flux": list(result.flux),
                "source": "k2_tess_overlap",
                "period_days": result.period_days,
                "epoch_bjd": result.epoch_bjd,
                "n_bins": result.n_bins,
                "epic_id": result.epic_id,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            n_written += 1
            n_errors = 0
            _emit_progress(
                f"  [{n_processed}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                f" EPIC {result.epic_id} P={result.period_days:.4f}d"
                f" label={result.label} {group_progress}"
                f"  written={n_written}  elapsed={elapsed:.0f}s  ETA={eta}"
            )
        else:
            n_skipped += 1
            n_errors += 1
            if result.flag in _TERMINAL_FAILURE_FLAGS:
                failure_record = {
                    "epic_id": result.epic_id,
                    "label": result.label,
                    "source": "k2_tess_overlap",
                    "period_days": result.period_days,
                    "epoch_bjd": result.epoch_bjd,
                    "n_bins": result.n_bins,
                    "flag": result.flag,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
                failure_fh.write(json.dumps(failure_record) + "\n")
                failure_fh.flush()
                n_terminal_failures += 1
            _emit_progress(
                f"  [{n_processed}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                f" EPIC {result.epic_id} P={result.period_days:.4f}d"
                f" SKIP flag={result.flag} {group_progress}"
                f"  errors={n_errors}  elapsed={elapsed:.0f}s  ETA={eta}"
            )

    with (
        output_path.open("a", encoding="utf-8") as fh,
        failure_log_path.open("a", encoding="utf-8") as failure_fh,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        group_iter = iter(enumerate(pending_groups, 1))
        future_meta = {}

        def submit_next_group() -> bool:
            try:
                group_index, group_rows = next(group_iter)
            except StopIteration:
                return False
            if request_delay > 0 and future_meta:
                time.sleep(request_delay)
            future = executor.submit(
                _process_epic_group,
                group_rows,
                n_bins=n_bins,
                lc_fetcher=lc_fetcher,
            )
            future_meta[future] = (group_index, group_rows)
            return True

        for _ in range(min(workers, n_groups)):
            submit_next_group()

        while future_meta:
            for future in as_completed(list(future_meta)):
                group_index, group_rows = future_meta.pop(future)
                break
            group_size = len(group_rows)
            try:
                results = future.result()
            except Exception as exc:
                results = [
                    K2SnippetResult(
                        epic_id=row.epic_id,
                        label=1 if row.disposition == "CONFIRMED" else 0,
                        flux=(),
                        period_days=row.period_days,
                        epoch_bjd=row.epoch_bkjd + _K2_BJD_OFFSET,
                        n_bins=n_bins,
                        flag=f"ERROR:{exc}",
                    )
                    for row in group_rows
                ]
            for result in results:
                write_result(result, group_index, group_size)
                if n_errors >= max_errors:
                    _emit_progress(f"Stopping early: {n_errors} consecutive non-OK results.")
                    for pending_future in future_meta:
                        pending_future.cancel()
                    break
            if n_errors >= max_errors:
                break
            submit_next_group()

    elapsed_total = time.monotonic() - start
    _emit_progress(
        f"Done. wrote={n_written} skipped={n_skipped} "
        f"terminal_failures_recorded={n_terminal_failures} "
        f"total_elapsed={elapsed_total:.0f}s"
    )
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_tess_k2_overlap_snippets",
        description=(
            "Fetch TESS light curves for K2 EPIC stars and write phase-folded"
            " CNN snippets to a JSONL file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tess_k2_overlap_snippets.jsonl"),
        metavar="PATH",
        help="Output JSONL file (default: data/tess_k2_overlap_snippets.jsonl)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=201,
        metavar="N",
        help="Number of phase bins per snippet (default: 201)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        metavar="N",
        help="Stop after N consecutive non-OK results (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of EPIC groups to fetch concurrently (default: 4)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Delay between submitting EPIC fetch jobs (default: 0.25)",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="Terminal failure JSONL sidecar (default: <output>.failures.jsonl)",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Recheck rows recorded in the terminal failure sidecar.",
    )
    parser.add_argument(
        "--k2-url",
        default=None,
        metavar="URL",
        help="Override the full NASA TAP URL (skips schema discovery; for testing)",
    )
    args = parser.parse_args(argv)

    _emit_progress(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching K2 table from NASA TAP...")
    rows = fetch_k2_table(args.k2_url)
    _emit_progress(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(rows)} K2 rows"
        f" ({sum(1 for r in rows if r.disposition == 'CONFIRMED')} CONFIRMED,"
        f" {sum(1 for r in rows if r.disposition == 'FALSE POSITIVE')} FALSE POSITIVE)."
    )

    n = build_k2_tess_snippets(
        rows,
        n_bins=args.n_bins,
        output_path=args.output,
        max_errors=args.max_errors,
        failure_log_path=args.failure_log,
        retry_failures=args.retry_failures,
        workers=args.workers,
        request_delay=args.request_delay,
    )
    _emit_progress(f"[{datetime.now().strftime('%H:%M:%S')}] Wrote {n} snippets -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
