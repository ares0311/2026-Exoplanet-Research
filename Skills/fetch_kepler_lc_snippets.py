"""Fetch Kepler KOI light curves and build phase-folded CNN snippets.

Downloads confirmed planets (CONFIRMED, label=1) and false positives
(FALSE POSITIVE, label=0) from the NASA Exoplanet Archive KOI cumulative
table, then fetches Kepler PDC-SAP light curves via lightkurve, phase-folds
at the KOI period/epoch, bins to ``n_bins`` phase bins, and normalises.

Output is a JSONL file (one JSON object per line) in the same format as
``data/tess_snippets.jsonl``::

    {"tic_id": 0, "kepid": 757450, "label": 1, "flux": [...], "source": "kepler",
     "period_days": 2.204, "epoch_bjd": 2454833.0 + koi_time0bk, "n_bins": 201}

Public API
----------
KoiSnippetResult(kepid, label, flux, period_days, epoch_bjd, n_bins, flag)
fetch_koi_table(max_rows) -> list[dict]
build_kepler_snippet(kepid, label, period_days, epoch_bjd, *, n_bins,
                     lc_fetcher) -> KoiSnippetResult
build_kepler_snippets(koi_rows, *, n_bins, output_path, lc_fetcher,
                      resume, max_errors, workers) -> int
"""
from __future__ import annotations

import contextlib
import json
import socket
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

# Prevent indefinite hangs when WiFi drops mid-download.
# Any stalled socket operation raises socket.timeout after this many seconds,
# which build_kepler_snippet's try/except catches and marks as ERROR.
socket.setdefaulttimeout(120)

_KEPLER_BJD_OFFSET = 2454833.0  # Kepler time is BJD - 2454833


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KoiSnippetResult:
    """Outcome of processing a single KOI."""

    kepid: int
    label: int            # 1 = confirmed planet, 0 = false positive
    flux: tuple[float, ...]
    period_days: float
    epoch_bjd: float
    n_bins: int
    flag: str             # "OK" | "NO_LIGHTKURVE" | "NO_DATA" | "SHORT" | "ERROR"


@dataclass(frozen=True)
class KoiTask:
    """Validated KOI row ready for light-curve fetching and phase folding."""

    row_index: int
    kepid: int
    kepoi_name: str
    label: int
    period_days: float
    epoch_bjd: float
    n_bins: int


@dataclass(frozen=True)
class KicGroupResult:
    """Result from processing all KOIs for one KIC light curve."""

    kepid: int
    row_indices: tuple[int, ...]
    records: tuple[dict, ...]
    flags: tuple[str, ...]


# ---------------------------------------------------------------------------
# Phase-fold and normalise helpers (mirror labelled_lc_collector.py)
# ---------------------------------------------------------------------------


def _phase_fold_bin(
    time_bjd: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> list[float]:
    """Phase-fold and median-bin a light curve into n_bins uniform bins."""
    phases = [((t - epoch) % period) / period for t in time_bjd]
    phases = [p - 1.0 if p >= 0.5 else p for p in phases]

    bin_flux: list[list[float]] = [[] for _ in range(n_bins)]
    for ph, f in zip(phases, flux, strict=False):
        b = int((ph + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bin_flux[b].append(f)
    return [
        sum(vals) / len(vals) if vals else 1.0
        for vals in bin_flux
    ]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mad(values: list[float], med: float) -> float:
    devs = [abs(v - med) for v in values]
    return _median(devs)


def _normalise(flux_bins: list[float]) -> list[float]:
    """Median/MAD normalisation (same as TESS snippet normaliser)."""
    med = _median(flux_bins)
    mad = _mad(flux_bins, med)
    scale = mad * 1.4826
    if scale < 1e-10:
        return [0.0] * len(flux_bins)
    return [(v - med) / scale for v in flux_bins]


# ---------------------------------------------------------------------------
# KOI table fetcher (NASA Exoplanet Archive TAP)
# ---------------------------------------------------------------------------

_NEA_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
)

_KOI_QUERY = (
    "SELECT kepid, kepoi_name, koi_disposition, koi_pdisposition, "
    "koi_period, koi_time0bk "
    "FROM cumulative "
    "WHERE (koi_disposition='CONFIRMED' OR koi_disposition='FALSE POSITIVE') "
    "AND koi_period > 0.5 AND koi_period < 500 "
    "AND koi_time0bk > 0 "
    "ORDER BY kepid"
)


def fetch_koi_table(max_rows: int = 10000) -> list[dict]:
    """Fetch KOI cumulative table from NASA Exoplanet Archive.

    Args:
        max_rows: Maximum number of rows to return.

    Returns:
        List of dicts with keys: kepid, kepoi_name, koi_disposition,
        koi_pdisposition, koi_period, koi_time0bk.
    """
    import ssl
    try:
        import certifi
        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    params = urlencode({
        "QUERY": _KOI_QUERY + f" TOP {max_rows}",
        "FORMAT": "json",
    })
    url = f"{_NEA_TAP_URL}?{params}"
    with urlopen(url, timeout=60, context=ctx) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, list):
        return data
    return data.get("data", [])


# ---------------------------------------------------------------------------
# Lightkurve fetcher helper
# ---------------------------------------------------------------------------


def _default_lc_fetcher(
    kepid: int,
    period: float,
    epoch_bjd: float,
) -> tuple[list[float], list[float]] | None:
    """Fetch a Kepler PDCSAP light curve for kepid.

    Returns:
        (time_bjd, flux) lists or None if unavailable.
    """
    try:
        import lightkurve as lk
    except ImportError:
        return None

    result = lk.search_lightcurve(
        f"KIC {kepid}", mission="Kepler", exptime=1800, author="Kepler"
    )
    if len(result) == 0:
        return None

    lc_coll = result.download_all()
    if lc_coll is None or len(lc_coll) == 0:
        return None

    lc = lc_coll.stitch()
    with contextlib.suppress(Exception):
        lc = lc.normalize()

    time_bjd = [float(t) + _KEPLER_BJD_OFFSET for t in lc.time.value]
    flux = [float(f) for f in lc.flux.value]
    return time_bjd, flux


# ---------------------------------------------------------------------------
# Per-KOI snippet builder
# ---------------------------------------------------------------------------


def build_kepler_snippet(
    kepid: int,
    label: int,
    period_days: float,
    epoch_bjd: float,
    *,
    n_bins: int = 201,
    lc_fetcher: Callable | None = None,
) -> KoiSnippetResult:
    """Build a single phase-folded CNN snippet for a Kepler KOI.

    Args:
        kepid: Kepler Input Catalog ID.
        label: 1 = confirmed planet, 0 = false positive.
        period_days: Orbital period in days.
        epoch_bjd: Transit epoch in BJD (full BJD, not Kepler offset).
        n_bins: Number of phase bins.
        lc_fetcher: Injectable fetcher returning (time_bjd, flux) or None.

    Returns:
        :class:`KoiSnippetResult` with flag "OK" on success.
    """
    fetcher = lc_fetcher or _default_lc_fetcher

    try:
        result = fetcher(kepid, period_days, epoch_bjd)
    except Exception as exc:
        return KoiSnippetResult(
            kepid=kepid, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag=f"ERROR:{exc}",
        )

    if result is None:
        try:
            import lightkurve  # noqa: F401
        except ImportError:
            return KoiSnippetResult(
                kepid=kepid, label=label, flux=(), period_days=period_days,
                epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_LIGHTKURVE",
            )
        return KoiSnippetResult(
            kepid=kepid, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_DATA",
        )

    time_bjd, flux = result
    if len(time_bjd) < n_bins:
        return KoiSnippetResult(
            kepid=kepid, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="SHORT",
        )

    bins = _phase_fold_bin(time_bjd, flux, period_days, epoch_bjd, n_bins)
    normalised = _normalise(bins)

    return KoiSnippetResult(
        kepid=kepid,
        label=label,
        flux=tuple(normalised),
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        n_bins=n_bins,
        flag="OK",
    )


def _task_from_row(row: dict, row_index: int, n_bins: int) -> KoiTask | None:
    try:
        kepid = int(row.get("kepid", 0))
        disposition = str(row.get("koi_disposition", ""))
        period = float(row.get("koi_period", 0.0))
        time0bk = float(row.get("koi_time0bk", 0.0))
    except (TypeError, ValueError):
        return None
    epoch_bjd = time0bk + _KEPLER_BJD_OFFSET
    if kepid <= 0 or period <= 0 or epoch_bjd <= _KEPLER_BJD_OFFSET:
        return None
    return KoiTask(
        row_index=row_index,
        kepid=kepid,
        kepoi_name=str(row.get("kepoi_name", "")),
        label=1 if disposition == "CONFIRMED" else 0,
        period_days=period,
        epoch_bjd=epoch_bjd,
        n_bins=n_bins,
    )


def _task_key(task: KoiTask) -> tuple[int, float, float, int]:
    return (
        task.kepid,
        round(task.period_days, 12),
        round(task.epoch_bjd, 8),
        task.label,
    )


def _completed_keys(output_path: Path) -> set[tuple[int, float, float, int]]:
    completed: set[tuple[int, float, float, int]] = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(
                    (
                        int(row.get("kepid", -1)),
                        round(float(row.get("period_days", 0.0)), 12),
                        round(float(row.get("epoch_bjd", 0.0)), 8),
                        int(row.get("label", -1)),
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return completed


def _group_tasks_by_kic(tasks: list[KoiTask]) -> list[list[KoiTask]]:
    groups: dict[int, list[KoiTask]] = {}
    for task in tasks:
        groups.setdefault(task.kepid, []).append(task)
    return [
        sorted(group, key=lambda task: task.row_index)
        for _kepid, group in sorted(groups.items())
    ]


def _record_from_result(result: KoiSnippetResult, task: KoiTask) -> dict:
    return {
        "tic_id": 0,
        "kepid": task.kepid,
        "kepoi_name": task.kepoi_name,
        "label": task.label,
        "flux": list(result.flux),
        "source": "kepler",
        "period_days": task.period_days,
        "epoch_bjd": task.epoch_bjd,
        "n_bins": task.n_bins,
    }


def _result_from_curve(
    task: KoiTask,
    time_bjd: list[float],
    flux: list[float],
) -> KoiSnippetResult:
    if len(time_bjd) < task.n_bins:
        return KoiSnippetResult(
            kepid=task.kepid, label=task.label, flux=(),
            period_days=task.period_days, epoch_bjd=task.epoch_bjd,
            n_bins=task.n_bins, flag="SHORT",
        )
    bins = _phase_fold_bin(
        time_bjd,
        flux,
        task.period_days,
        task.epoch_bjd,
        task.n_bins,
    )
    return KoiSnippetResult(
        kepid=task.kepid,
        label=task.label,
        flux=tuple(_normalise(bins)),
        period_days=task.period_days,
        epoch_bjd=task.epoch_bjd,
        n_bins=task.n_bins,
        flag="OK",
    )


def _process_kic_group(
    tasks: list[KoiTask],
    *,
    lc_fetcher: Callable | None,
) -> KicGroupResult:
    task0 = tasks[0]
    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        fetched = fetcher(task0.kepid, task0.period_days, task0.epoch_bjd)
    except Exception as exc:
        return KicGroupResult(
            kepid=task0.kepid,
            row_indices=tuple(task.row_index for task in tasks),
            records=(),
            flags=tuple(f"ERROR:{exc}" for _task in tasks),
        )

    if fetched is None:
        try:
            import lightkurve  # noqa: F401
        except ImportError:
            flag = "NO_LIGHTKURVE"
        else:
            flag = "NO_DATA"
        return KicGroupResult(
            kepid=task0.kepid,
            row_indices=tuple(task.row_index for task in tasks),
            records=(),
            flags=tuple(flag for _task in tasks),
        )

    time_bjd, flux = fetched
    records: list[dict] = []
    flags: list[str] = []
    for task in tasks:
        result = _result_from_curve(task, time_bjd, flux)
        flags.append(result.flag)
        if result.flag == "OK":
            records.append(_record_from_result(result, task))
    return KicGroupResult(
        kepid=task0.kepid,
        row_indices=tuple(task.row_index for task in tasks),
        records=tuple(records),
        flags=tuple(flags),
    )


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def build_kepler_snippets(
    koi_rows: list[dict],
    *,
    n_bins: int = 201,
    output_path: Path,
    lc_fetcher: Callable | None = None,
    resume: bool = True,
    max_errors: int = 50,
    workers: int = 1,
    request_delay: float = 0.25,
) -> int:
    """Build phase-folded CNN snippets for a list of KOI rows.

    Writes one JSONL record per successful snippet. Skips already-written KOI
    signatures when ``resume=True``. Multiple KOIs around the same KIC share
    one light-curve fetch, then fold locally at each KOI period/epoch.

    Args:
        koi_rows: List of dicts (from :func:`fetch_koi_table`).
        n_bins: Number of phase bins per snippet.
        output_path: Destination JSONL file path.
        lc_fetcher: Injectable light-curve fetcher.
        resume: Skip KOI signatures already present in output_path.
        max_errors: Abort after this many consecutive failed KIC groups.
        workers: Bounded thread count for concurrent KIC fetches.
        request_delay: Delay between worker submissions when ``workers > 1``.

    Returns:
        Number of snippets written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _completed_keys(output_path) if resume else set()
    tasks = [
        task
        for index, row in enumerate(koi_rows, 1)
        if (task := _task_from_row(row, index, n_bins)) is not None
    ]
    pending_tasks = [task for task in tasks if not (resume and _task_key(task) in completed)]
    task_groups = _group_tasks_by_kic(pending_tasks)
    n_total = len(tasks)
    n_written = 0
    n_errors = 0
    consecutive_errors = 0
    start = time.monotonic()
    workers = max(1, int(workers))

    print(
        f"Building Kepler snippets: {n_total} valid KOIs  resume={resume}  "
        f"already_done={len(completed)}  pending={len(pending_tasks)}  "
        f"kic_groups={len(task_groups)}  workers={workers}  output={output_path}",
        flush=True,
    )

    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as fh:
        if workers == 1:
            for group_index, group in enumerate(task_groups, 1):
                group_result = _process_kic_group(group, lc_fetcher=lc_fetcher)
                wrote, failed = _write_group_result(fh, group_result)
                n_written += wrote
                n_errors += failed
                consecutive_errors = 0 if wrote else consecutive_errors + failed
                _print_group_progress(
                    group_index,
                    len(task_groups),
                    group_result,
                    n_written,
                    n_errors,
                    start,
                )
                if consecutive_errors >= max_errors:
                    print(
                        f"Aborting: {consecutive_errors} consecutive KIC-group errors.",
                        flush=True,
                    )
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                in_flight = {}
                next_group_index = 1
                completed_groups = 0
                should_abort = False

                while next_group_index <= len(task_groups) and len(in_flight) < workers:
                    group = task_groups[next_group_index - 1]
                    future = executor.submit(
                        _process_kic_group,
                        group,
                        lc_fetcher=lc_fetcher,
                    )
                    in_flight[future] = next_group_index
                    next_group_index += 1
                    if request_delay > 0:
                        time.sleep(request_delay)

                while in_flight:
                    done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        in_flight.pop(future)
                        group_result = future.result()
                        completed_groups += 1
                        wrote, failed = _write_group_result(fh, group_result)
                        n_written += wrote
                        n_errors += failed
                        consecutive_errors = 0 if wrote else consecutive_errors + failed
                        _print_group_progress(
                            completed_groups,
                            len(task_groups),
                            group_result,
                            n_written,
                            n_errors,
                            start,
                        )
                        if consecutive_errors >= max_errors:
                            print(
                                f"Aborting: {consecutive_errors} consecutive KIC-group errors.",
                                flush=True,
                            )
                            should_abort = True
                            break
                    if should_abort:
                        for future in in_flight:
                            future.cancel()
                        break
                    while next_group_index <= len(task_groups) and len(in_flight) < workers:
                        group = task_groups[next_group_index - 1]
                        future = executor.submit(
                            _process_kic_group,
                            group,
                            lc_fetcher=lc_fetcher,
                        )
                        in_flight[future] = next_group_index
                        next_group_index += 1
                        if request_delay > 0:
                            time.sleep(request_delay)

    print(
        f"Done. {n_written} snippets written, {n_errors} skipped/errors.",
        flush=True,
    )
    return n_written


def _write_group_result(handle, group_result: KicGroupResult) -> tuple[int, int]:
    for record in group_result.records:
        handle.write(json.dumps(record) + "\n")
    if group_result.records:
        handle.flush()
    return len(group_result.records), sum(
        1 for flag in group_result.flags if flag != "OK"
    )


def _print_group_progress(
    group_index: int,
    n_groups: int,
    group_result: KicGroupResult,
    n_written: int,
    n_errors: int,
    start: float,
) -> None:
    elapsed = time.monotonic() - start
    rate = group_index / elapsed if elapsed > 0 else 1.0
    remaining = (n_groups - group_index) / rate
    eta = (
        f"{remaining/60:.0f}m{remaining%60:.0f}s"
        if remaining > 90
        else f"{remaining:.0f}s"
    )
    ok_count = len(group_result.records)
    fail_count = sum(1 for flag in group_result.flags if flag != "OK")
    flags = ",".join(sorted(set(group_result.flags))) if group_result.flags else "EMPTY"
    print(
        f"  [KIC {group_index}/{n_groups}] {datetime.now().strftime('%H:%M:%S')}"
        f" KIC {group_result.kepid} rows={len(group_result.row_indices)}"
        f" ok={ok_count} fail={fail_count} flags={flags}"
        f"  written={n_written} errors={n_errors}"
        f"  elapsed={elapsed:.0f}s  ETA={eta}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_kepler_lc_snippets",
        description=(
            "Fetch Kepler KOI light curves and build phase-folded CNN snippets."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/kepler_snippets.jsonl"),
        metavar="JSONL",
        help="Output JSONL file (appended when --resume; default: data/kepler_snippets.jsonl)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=201, metavar="N",
        help="Number of phase bins per snippet (default: 201)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=10000, metavar="N",
        help="Maximum KOI rows to fetch from NEA (default: 10000)",
    )
    parser.add_argument(
        "--max-errors", type=int, default=50, metavar="N",
        help="Abort after this many consecutive fetch errors (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help=(
            "Concurrent KIC light-curve fetches. Use 2-4 to speed up while "
            "staying polite to MAST (default: 1)."
        ),
    )
    parser.add_argument(
        "--request-delay", type=float, default=0.25, metavar="SEC",
        help=(
            "Delay between worker submissions when --workers > 1 "
            "(default: 0.25)."
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh even if output file exists",
    )
    args = parser.parse_args(argv)

    print("Fetching KOI table from NASA Exoplanet Archive ...", flush=True)
    try:
        koi_rows = fetch_koi_table(max_rows=args.max_rows)
    except Exception as exc:
        print(f"ERROR fetching KOI table: {exc}")
        return 1

    confirmed = sum(1 for r in koi_rows if r.get("koi_disposition") == "CONFIRMED")
    fp = sum(1 for r in koi_rows if r.get("koi_disposition") == "FALSE POSITIVE")
    print(
        f"KOI table: {len(koi_rows)} rows  CONFIRMED={confirmed}  FALSE POSITIVE={fp}",
        flush=True,
    )

    n = build_kepler_snippets(
        koi_rows,
        n_bins=args.n_bins,
        output_path=args.output,
        resume=not args.no_resume,
        max_errors=args.max_errors,
        workers=args.workers,
        request_delay=args.request_delay,
    )
    print(f"Flag: OK  snippets_written={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
