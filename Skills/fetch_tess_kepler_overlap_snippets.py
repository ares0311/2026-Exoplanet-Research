"""Fetch TESS light curves for Kepler KOI stars to build the overlap corpus.

Downloads phase-folded TESS snippets for all Kepler-confirmed planets and
confirmed false positives that MAST has also observed with TESS.  Intended for
expanding the TESS training corpus beyond what ExoFOP-TESS provides (which
yielded only 56 new labeled TIC IDs as of 2026-06-18).

The Kepler KOI ephemerides (period and epoch) are used to phase-fold the TESS
light curves.  The epoch is converted from BKJD (BJD − 2454833) to full BJD
before folding; TESS light curve timestamps (BTJD = BJD − 2457000) are also
converted to full BJD before folding.

Output format matches ``data/tess_snippets_v2.jsonl``::

    {"tic_id": <kepid>, "label": 0|1, "flux": [...201 floats...],
     "source": "koi_tess_overlap", "period_days": 3.4,
     "epoch_bjd": 2454900.0, "n_bins": 201, "kepoi_name": "K00001.01"}

The ``tic_id`` field stores the Kepler KIC ID for corpus compatibility; it does
**not** represent a TESS TIC ID.  Downstream tools (``build_cnn_training_data.py``)
only use ``flux`` and ``label``, so this is safe.  The ``kepoi_name`` field is
present for provenance only.

Resume is automatic: rows whose ``kepoi_name`` already appears in the output
JSONL are skipped, including any rows written in a prior interrupted run.
Terminal failures are recorded in ``<output>.failures.jsonl`` and skipped on
ordinary reruns; pass ``--retry-failures`` for an intentional recheck.

Run command (Mac only — requires .venv with lightkurve):
    caffeinate -dims .venv/bin/python Skills/fetch_tess_kepler_overlap_snippets.py \\
        --output data/tess_kepler_overlap_snippets.jsonl \\
        --workers 4 \\
        --request-delay 0.25

After download, merge with the existing TESS corpus and rebuild splits:
    cat data/tess_snippets_v2.jsonl data/tess_kepler_overlap_snippets.jsonl \\
        > data/tess_combined_snippets.jsonl
    .venv/bin/python Skills/build_cnn_training_data.py \\
        data/tess_combined_snippets.jsonl --output-dir data/tess_combined_cnn_splits --seed 7

Public API
----------
KoiRow(kepid, kepoi_name, disposition, period_days, epoch_bkjd)
KoiSnippetResult(kepid, kepoi_name, label, flux, period_days, epoch_bjd, n_bins, flag)
fetch_koi_table(url) -> list[KoiRow]
build_koi_tess_snippet(row, *, n_bins, lc_fetcher) -> KoiSnippetResult
build_koi_tess_snippets(rows, *, n_bins, output_path, lc_fetcher, max_errors,
                        workers, request_delay) -> int
"""
from __future__ import annotations

import contextlib
import json
import math
import socket
import ssl
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

# Prevent indefinite hangs when WiFi drops mid-download.
socket.setdefaulttimeout(120)

_KEPLER_BJD_OFFSET = 2454833.0  # koi_time0bk (BKJD) = BJD − 2454833
_TESS_BJD_OFFSET = 2457000.0    # TESS BTJD = BJD − 2457000

# NASA Exoplanet Archive TAP: all KOIs with confirmed or FP dispositions.
_KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk"
    "+from+cumulative"
    "+where+koi_disposition+in+('CONFIRMED','FALSE+POSITIVE')"
    "+and+koi_period+is+not+null"
    "+and+koi_time0bk+is+not+null"
    "&format=json"
)
_TERMINAL_FAILURE_FLAGS = {"NO_LIGHTKURVE", "NO_DATA", "SHORT", "NONFINITE"}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KoiRow:
    """A single KOI record from the NASA Exoplanet Archive cumulative table."""

    kepid: int
    kepoi_name: str
    disposition: str   # "CONFIRMED" or "FALSE POSITIVE"
    period_days: float
    epoch_bkjd: float  # BJD − 2454833


@dataclass(frozen=True)
class KoiSnippetResult:
    """Outcome of processing a single KOI against TESS data."""

    kepid: int
    kepoi_name: str
    label: int            # 1 = CONFIRMED, 0 = FALSE POSITIVE
    flux: tuple[float, ...]
    period_days: float
    epoch_bjd: float
    n_bins: int
    flag: str             # "OK" | "NO_LIGHTKURVE" | "NO_DATA" | "SHORT" | "NONFINITE" | "ERROR:..."


# ---------------------------------------------------------------------------
# Phase-fold / normalise helpers (identical to fetch_tess_lc_snippets.py)
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
# KOI table fetcher
# ---------------------------------------------------------------------------


def fetch_koi_table(url: str = _KOI_TAP_URL) -> list[KoiRow]:
    """Fetch the KOI cumulative table from the NASA Exoplanet Archive TAP service.

    Args:
        url: TAP URL (overridable for tests).

    Returns:
        List of :class:`KoiRow` objects with valid period and epoch.

    Raises:
        RuntimeError: If the table cannot be fetched or parsed.
    """
    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None

    with urlopen(url, timeout=120, context=ctx) as resp:  # noqa: S310
        raw = json.loads(resp.read())

    rows: list[KoiRow] = []
    for rec in raw:
        try:
            kepid = int(rec["kepid"])
            kepoi_name = str(rec["kepoi_name"])
            disposition = str(rec["koi_disposition"]).strip().upper()
            period = float(rec["koi_period"])
            epoch_bkjd = float(rec["koi_time0bk"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(period) or period <= 0:
            continue
        if not math.isfinite(epoch_bkjd):
            continue
        if disposition not in {"CONFIRMED", "FALSE POSITIVE"}:
            continue
        rows.append(KoiRow(
            kepid=kepid,
            kepoi_name=kepoi_name,
            disposition=disposition,
            period_days=period,
            epoch_bkjd=epoch_bkjd,
        ))
    return rows


# ---------------------------------------------------------------------------
# TESS light curve fetcher for a KIC star
# ---------------------------------------------------------------------------


def _default_lc_fetcher(
    kepid: int,
    period: float,
    epoch_bjd: float,
) -> tuple[list[float], list[float]] | None:
    """Fetch a TESS light curve for a Kepler star (KIC ID) via lightkurve.

    Searches MAST for TESS SPOC or QLP data for "KIC {kepid}".  MAST resolves
    KIC identifiers to sky coordinates and returns any TESS sectors that covered
    the same position.

    Returns:
        (time_bjd, flux) in full BJD, or None if no TESS data found.
    """
    try:
        import lightkurve as lk
    except ImportError:
        return None

    for author in ("SPOC", "QLP"):
        result = lk.search_lightcurve(f"KIC {kepid}", mission="TESS", author=author)
        if len(result) == 0:
            continue
        lc_coll = result.download_all()
        if lc_coll is None or len(lc_coll) == 0:
            continue
        lc = lc_coll.stitch()
        with contextlib.suppress(Exception):
            lc = lc.normalize()
        # TESS time is BTJD = BJD - 2457000; convert to full BJD.
        time_bjd = [float(t) + _TESS_BJD_OFFSET for t in lc.time.value]
        flux = [float(f) for f in lc.flux.value]
        return time_bjd, flux

    return None


# ---------------------------------------------------------------------------
# Per-KOI snippet builder
# ---------------------------------------------------------------------------


def build_koi_tess_snippet(
    row: KoiRow,
    *,
    n_bins: int = 201,
    lc_fetcher: Callable | None = None,
) -> KoiSnippetResult:
    """Build a single phase-folded TESS snippet for a KOI using its Kepler ephemeris.

    Args:
        row: KOI record with period and epoch in BKJD.
        n_bins: Number of phase bins for the output snippet.
        lc_fetcher: Injectable fetcher returning (time_bjd, flux) or None.

    Returns:
        :class:`KoiSnippetResult` with flag "OK" on success.
    """
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _KEPLER_BJD_OFFSET

    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        raw = fetcher(row.kepid, row.period_days, epoch_bjd)
    except Exception as exc:
        return KoiSnippetResult(
            kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag=f"ERROR:{exc}",
        )

    if raw is None:
        return _missing_light_curve_result(row, n_bins=n_bins)

    return _build_koi_tess_snippet_from_raw(row, raw, n_bins=n_bins)


def _missing_light_curve_result(row: KoiRow, *, n_bins: int) -> KoiSnippetResult:
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _KEPLER_BJD_OFFSET
    try:
        import lightkurve  # noqa: F401
    except ImportError:
        return KoiSnippetResult(
            kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="NO_LIGHTKURVE",
        )
    return KoiSnippetResult(
        kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
        flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
        n_bins=n_bins, flag="NO_DATA",
    )


def _build_koi_tess_snippet_from_raw(
    row: KoiRow,
    raw: tuple[list[float], list[float]],
    *,
    n_bins: int,
) -> KoiSnippetResult:
    label = 1 if row.disposition == "CONFIRMED" else 0
    epoch_bjd = row.epoch_bkjd + _KEPLER_BJD_OFFSET

    time_bjd, flux = raw
    finite_pairs = [
        (t, f)
        for t, f in zip(time_bjd, flux, strict=False)
        if math.isfinite(t) and math.isfinite(f)
    ]
    if len(finite_pairs) < n_bins:
        return KoiSnippetResult(
            kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="SHORT",
        )

    t_bjd = [p[0] for p in finite_pairs]
    f_vals = [p[1] for p in finite_pairs]
    bins = _phase_fold_bin(t_bjd, f_vals, row.period_days, epoch_bjd, n_bins)
    normalised = _normalise(bins)
    if len(normalised) != n_bins or any(not math.isfinite(v) for v in normalised):
        return KoiSnippetResult(
            kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
            flux=(), period_days=row.period_days, epoch_bjd=epoch_bjd,
            n_bins=n_bins, flag="NONFINITE",
        )

    return KoiSnippetResult(
        kepid=row.kepid, kepoi_name=row.kepoi_name, label=label,
        flux=tuple(normalised), period_days=row.period_days,
        epoch_bjd=epoch_bjd, n_bins=n_bins, flag="OK",
    )


def _process_kepid_group(
    group_rows: list[KoiRow],
    *,
    n_bins: int,
    lc_fetcher: Callable | None,
) -> list[KoiSnippetResult]:
    """Fetch one TESS light curve for a KIC and fold each KOI locally."""
    first = group_rows[0]
    first_epoch_bjd = first.epoch_bkjd + _KEPLER_BJD_OFFSET
    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        raw = fetcher(first.kepid, first.period_days, first_epoch_bjd)
    except Exception as exc:
        return [
            KoiSnippetResult(
                kepid=row.kepid,
                kepoi_name=row.kepoi_name,
                label=1 if row.disposition == "CONFIRMED" else 0,
                flux=(),
                period_days=row.period_days,
                epoch_bjd=row.epoch_bkjd + _KEPLER_BJD_OFFSET,
                n_bins=n_bins,
                flag=f"ERROR:{exc}",
            )
            for row in group_rows
        ]
    if raw is None:
        return [_missing_light_curve_result(row, n_bins=n_bins) for row in group_rows]
    return [
        _build_koi_tess_snippet_from_raw(row, raw, n_bins=n_bins)
        for row in group_rows
    ]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def build_koi_tess_snippets(
    rows: list[KoiRow],
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
    """Build phase-folded TESS snippets for a list of KOI rows.

    Writes to ``output_path`` in append mode (``"a"``); existing content is
    read at startup to populate the ``already_done`` resume set.  The file is
    created if it does not exist.

    Args:
        rows: KOI records to process.
        n_bins: Number of phase bins per snippet.
        output_path: Path to the JSONL output file.
        lc_fetcher: Injectable fetcher for testing.
        max_errors: Stop early after this many consecutive non-OK results.
        failure_log_path: Terminal failure JSONL sidecar path.
        retry_failures: Reprocess rows recorded in the failure sidecar.
        workers: Number of KIC groups to process concurrently.
        request_delay: Seconds to wait between submitting KIC fetch jobs.

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
                    kn = obj.get("kepoi_name")
                    if kn:
                        already_done.add(str(kn))
                except json.JSONDecodeError:
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
                    kn = obj.get("kepoi_name")
                    flag = obj.get("flag")
                    if kn and flag in _TERMINAL_FAILURE_FLAGS:
                        already_failed.add(str(kn))
                except json.JSONDecodeError:
                    pass

    pending_rows = [
        r for r in rows
        if r.kepoi_name not in already_done and r.kepoi_name not in already_failed
    ]
    pending_by_kepid: dict[int, list[KoiRow]] = defaultdict(list)
    for row in pending_rows:
        pending_by_kepid[row.kepid].append(row)
    pending_groups = list(pending_by_kepid.values())
    n_total = len(pending_rows)
    n_groups = len(pending_groups)
    n_processed = 0
    n_written = 0
    n_skipped = 0
    n_terminal_failures = 0
    n_errors = 0
    start = time.monotonic()

    print(
        f"Kepler-TESS overlap fetch: {len(rows)} total KOIs, "
        f"{len(already_done)} already done, "
        f"{len(already_failed)} terminal failures skipped, "
        f"{n_total} pending across {n_groups} KIC groups.",
        flush=True,
    )
    print(
        f"Output: {output_path}  failures={failure_log_path}  "
        f"n_bins={n_bins}  max_errors={max_errors}  "
        f"workers={workers}  request_delay={request_delay:.2f}s  "
        f"retry_failures={retry_failures}",
        flush=True,
    )

    def write_result(result: KoiSnippetResult, group_index: int, group_size: int) -> None:
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
                "tic_id": result.kepid,
                "label": result.label,
                "flux": list(result.flux),
                "source": "koi_tess_overlap",
                "period_days": result.period_days,
                "epoch_bjd": result.epoch_bjd,
                "n_bins": result.n_bins,
                "kepoi_name": result.kepoi_name,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            n_written += 1
            n_errors = 0
            print(
                f"  [{n_processed}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                f" {result.kepoi_name} KIC {result.kepid}"
                f" label={result.label} {group_progress}"
                f"  written={n_written}  elapsed={elapsed:.0f}s  ETA={eta}",
                flush=True,
            )
        else:
            n_skipped += 1
            n_errors += 1
            if result.flag in _TERMINAL_FAILURE_FLAGS:
                failure_record = {
                    "kepoi_name": result.kepoi_name,
                    "kepid": result.kepid,
                    "label": result.label,
                    "source": "koi_tess_overlap",
                    "period_days": result.period_days,
                    "epoch_bjd": result.epoch_bjd,
                    "n_bins": result.n_bins,
                    "flag": result.flag,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
                failure_fh.write(json.dumps(failure_record) + "\n")
                failure_fh.flush()
                n_terminal_failures += 1
            print(
                f"  [{n_processed}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                f" {result.kepoi_name} KIC {result.kepid}"
                f" SKIP flag={result.flag} {group_progress}"
                f"  errors={n_errors}  elapsed={elapsed:.0f}s  ETA={eta}",
                flush=True,
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
                _process_kepid_group,
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
                    KoiSnippetResult(
                        kepid=row.kepid,
                        kepoi_name=row.kepoi_name,
                        label=1 if row.disposition == "CONFIRMED" else 0,
                        flux=(),
                        period_days=row.period_days,
                        epoch_bjd=row.epoch_bkjd + _KEPLER_BJD_OFFSET,
                        n_bins=n_bins,
                        flag=f"ERROR:{exc}",
                    )
                    for row in group_rows
                ]
            for result in results:
                write_result(result, group_index, group_size)
                if n_errors >= max_errors:
                    print(
                        f"Stopping early: {n_errors} consecutive non-OK results.",
                        flush=True,
                    )
                    for pending_future in future_meta:
                        pending_future.cancel()
                    break
            if n_errors >= max_errors:
                break
            submit_next_group()

    elapsed_total = time.monotonic() - start
    print(
        f"Done. wrote={n_written} skipped={n_skipped} "
        f"terminal_failures_recorded={n_terminal_failures} "
        f"total_elapsed={elapsed_total:.0f}s",
        flush=True,
    )
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_tess_kepler_overlap_snippets",
        description=(
            "Fetch TESS light curves for Kepler KOI stars and write phase-folded"
            " CNN snippets to a JSONL file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tess_kepler_overlap_snippets.jsonl"),
        metavar="PATH",
        help="Output JSONL file (default: data/tess_kepler_overlap_snippets.jsonl)",
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
        help="Number of KIC groups to fetch concurrently (default: 4)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Delay between submitting KIC fetch jobs (default: 0.25)",
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
        "--koi-url",
        default=_KOI_TAP_URL,
        metavar="URL",
        help="Override the NASA TAP URL for the KOI table",
    )
    args = parser.parse_args(argv)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Fetching KOI table from NASA TAP...",
        flush=True,
    )
    rows = fetch_koi_table(args.koi_url)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(rows)} KOI rows"
        f" ({sum(1 for r in rows if r.disposition == 'CONFIRMED')} CONFIRMED,"
        f" {sum(1 for r in rows if r.disposition == 'FALSE POSITIVE')} FALSE POSITIVE).",
        flush=True,
    )

    n = build_koi_tess_snippets(
        rows,
        n_bins=args.n_bins,
        output_path=args.output,
        max_errors=args.max_errors,
        failure_log_path=args.failure_log,
        retry_failures=args.retry_failures,
        workers=args.workers,
        request_delay=args.request_delay,
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Wrote {n} snippets → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
