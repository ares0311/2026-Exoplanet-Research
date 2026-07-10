"""Fetch native K2 light curves for the T1-2 held-out calibration manifest.

Reads ``metadata/t1_2_k2_calibration_manifest.jsonl`` (built by
``build_t1_2_k2_calibration_manifest.py``) and fetches the **native K2**
light curve (mission="K2", not TESS) for every EPIC target, phase-folding at
the manifest's period/epoch to a 201-bin normalised flux snippet — the input
format ``exo_toolkit.ml.cnn_scorer.CnnScorer`` expects.

Uses ``exo_toolkit.fetch.fetch_lightcurve()`` (the same production,
cache-repair-aware, retry-aware fetch path used by
``process_t1_kepler_batch.py``), not a raw ``lightkurve.download_all()``
call — that older pattern is unsafe under concurrent workers because it
mutates process-global ``sys.stdout`` (see ``docs/PRODUCTION_READINESS.md``
T1-0 run003 and the T1-1 version-0.2.17 fix for the same failure class).

K2's own light-curve time convention (via lightkurve) is raw BKJD
(BJD - 2454833), the same as Kepler prime-mission products -- NOT the same
as the ``pl_tranmid`` catalog column, which the manifest builder already
confirmed is full BJD. Do not conflate the two: this script adds the BKJD
offset only to the *light curve's* time array, never to the manifest's
already-full-BJD ``epoch_bjd``.

Resume is automatic: rows whose ``(epic_id, period_days)`` key already
appears in the output JSONL are skipped. Terminal failures are recorded in
``<output>.failures.jsonl`` and skipped on ordinary reruns; pass
``--retry-failures`` for an intentional recheck.

Run command (Mac only -- requires .venv with lightkurve; this fetches ~600
real light curves and should run under caffeinate per the macOS
long-running-process policy):
    caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py \\
        --manifest metadata/t1_2_k2_calibration_manifest.jsonl \\
        --output data/t1_2_k2_calibration_snippets.jsonl \\
        --workers 4

Also supports process-level sharding (``--shard-index``/``--shard-count``),
mirroring ``process_t1_kepler_batch.py``'s pattern, for running several
console tabs concurrently against disjoint target sets. Every concurrently
running tab must share the same ``--shard-count`` (each tab passes its own
unique ``--shard-index`` from ``0`` to ``shard_count - 1``). Partitioning is
by ``epic_id % shard_count``, which is disjoint across shards, so no two
tabs ever touch the same target. Each shard writes to its own auto-suffixed
output file (and failure log) -- never share one ``--output`` path across
concurrent processes (concurrent appends to one file are not guaranteed
atomic); concatenate the shard files afterward.

Example -- 4 terminals, 6 workers each:
    # Terminal 1
    caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py \\
        --workers 6 --shard-index 0 --shard-count 4
    # Terminal 2
    caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py \\
        --workers 6 --shard-index 1 --shard-count 4
    # Terminal 3
    caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py \\
        --workers 6 --shard-index 2 --shard-count 4
    # Terminal 4
    caffeinate -dims .venv/bin/python Skills/fetch_t1_2_k2_calibration_snippets.py \\
        --workers 6 --shard-index 3 --shard-count 4

Public API
----------
K2CalibrationSnippetResult(epic_id, label, flux, period_days, epoch_bjd, n_bins, flag)
build_k2_calibration_snippet(row, *, n_bins, lc_fetcher) -> K2CalibrationSnippetResult
shard_output_path(output_path, shard_index, shard_count) -> Path
build_k2_calibration_snippets(manifest_path, *, output_path, n_bins, workers,
                              max_errors, lc_fetcher, failure_log_path,
                              retry_failures, shard_index, shard_count) -> int
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_report import (  # noqa: E402
    RunReport,
    report_path_for,
    run_and_commit_report,
)

# K2 shares Kepler's raw light-curve time convention (BJD - 2454833). This is
# distinct from and must not be confused with the manifest's epoch_bjd, which
# is already full BJD (verified live against the k2pandc catalog).
_KEPLER_BJD_OFFSET = 2454833.0

_TERMINAL_FAILURE_FLAGS = {"NO_LIGHTKURVE", "NO_DATA", "SHORT", "NONFINITE"}


def _safe_print(message: str) -> None:
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


@dataclass(frozen=True)
class K2CalibrationSnippetResult:
    """Outcome of building one native K2 calibration snippet."""

    epic_id: int
    label: int
    flux: tuple[float, ...]
    period_days: float
    epoch_bjd: float
    n_bins: int
    flag: str  # "OK" | "NO_LIGHTKURVE" | "NO_DATA" | "SHORT" | "NONFINITE" | "ERROR:..."


# ---------------------------------------------------------------------------
# Phase-fold / normalise helpers (identical to fetch_tess_k2_overlap_snippets.py
# and fetch_kepler_lc_snippets.py -- this project copies these small, stable
# helpers per-fetcher rather than sharing a common module)
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
# Light curve fetcher
# ---------------------------------------------------------------------------


def _default_lc_fetcher(epic_id: int) -> tuple[list[float], list[float]] | None:
    """Fetch a native K2 light curve for one EPIC target.

    Uses ``exo_toolkit.fetch.fetch_lightcurve`` -- the same production,
    cache-repair-aware, transient-retry-aware path used elsewhere in this
    project -- rather than a raw ``lightkurve.download_all()`` call.

    Returns:
        ``(time_bjd, flux)`` in full BJD, or ``None`` if no K2 light curve
        was found for this target.

    Raises:
        Exception: Any non-"no data" error (network, parse, etc.) propagates
            so the caller can record it as a distinct ``ERROR:`` flag.
    """
    try:
        from exo_toolkit.fetch import fetch_lightcurve
    except ImportError:
        return None

    try:
        result = fetch_lightcurve(f"EPIC {epic_id}", "K2")
    except ValueError:
        return None  # no light curve found -- not an error, just no data

    lc = result.light_curve
    time_bjd = [float(t) + _KEPLER_BJD_OFFSET for t in lc.time.value]
    flux = [float(f) for f in lc.flux.value]
    return time_bjd, flux


# ---------------------------------------------------------------------------
# Per-target snippet builder
# ---------------------------------------------------------------------------


def build_k2_calibration_snippet(
    row: dict[str, Any],
    *,
    n_bins: int = 201,
    lc_fetcher: Callable[[int], tuple[list[float], list[float]] | None] | None = None,
) -> K2CalibrationSnippetResult:
    """Build one phase-folded, normalised native-K2 calibration snippet.

    Args:
        row: One parsed row from ``t1_2_k2_calibration_manifest.jsonl``
            (needs ``epic_id``, ``label``, ``period_days``, ``epoch_bjd``).
        n_bins: Number of phase bins for the output snippet.
        lc_fetcher: Injectable fetcher returning ``(time_bjd, flux)`` or
            ``None`` (for tests). Defaults to :func:`_default_lc_fetcher`.

    Returns:
        :class:`K2CalibrationSnippetResult` with ``flag="OK"`` on success.
    """
    epic_id = int(row["epic_id"])
    label = int(row["label"])
    period_days = float(row["period_days"])
    epoch_bjd = float(row["epoch_bjd"])

    fetcher = lc_fetcher or _default_lc_fetcher
    try:
        raw = fetcher(epic_id)
    except ImportError:
        return K2CalibrationSnippetResult(
            epic_id=epic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_LIGHTKURVE",
        )
    except Exception as exc:  # noqa: BLE001 -- recorded as a distinct terminal flag
        return K2CalibrationSnippetResult(
            epic_id=epic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag=f"ERROR:{exc}",
        )

    if raw is None:
        return K2CalibrationSnippetResult(
            epic_id=epic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_DATA",
        )

    time_bjd, flux = raw
    finite_pairs = [
        (t, f) for t, f in zip(time_bjd, flux, strict=False)
        if math.isfinite(t) and math.isfinite(f)
    ]
    if len(finite_pairs) < n_bins:
        return K2CalibrationSnippetResult(
            epic_id=epic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="SHORT",
        )

    t_vals = [t for t, _ in finite_pairs]
    f_vals = [f for _, f in finite_pairs]
    bins = _phase_fold_bin(t_vals, f_vals, period_days, epoch_bjd, n_bins)
    normalised = _normalise(bins)
    if not normalised:
        return K2CalibrationSnippetResult(
            epic_id=epic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NONFINITE",
        )

    return K2CalibrationSnippetResult(
        epic_id=epic_id, label=label, flux=tuple(normalised),
        period_days=period_days, epoch_bjd=epoch_bjd, n_bins=n_bins, flag="OK",
    )


# ---------------------------------------------------------------------------
# Resume plumbing
# ---------------------------------------------------------------------------


def _task_key(row: dict[str, Any]) -> tuple[int, float]:
    return (int(row["epic_id"]), round(float(row["period_days"]), 8))


def _completed_keys(output_path: Path) -> set[tuple[int, float]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[int, float]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        keys.add((int(rec["epic_id"]), round(float(rec["period_days"]), 8)))
    return keys


def _terminal_failure_keys(failure_log_path: Path) -> set[tuple[int, float]]:
    if not failure_log_path.exists():
        return set()
    keys: set[tuple[int, float]] = set()
    for line in failure_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        keys.add((int(rec["epic_id"]), round(float(rec["period_days"]), 8)))
    return keys


def _record_from_result(result: K2CalibrationSnippetResult) -> dict[str, Any]:
    return {
        "epic_id": result.epic_id,
        "label": result.label,
        "flux": list(result.flux),
        "period_days": result.period_days,
        "epoch_bjd": result.epoch_bjd,
        "n_bins": result.n_bins,
        "source": "k2_native_calibration",
    }


def _failure_record(row: dict[str, Any], flag: str) -> dict[str, Any]:
    return {
        "epic_id": int(row["epic_id"]),
        "period_days": float(row["period_days"]),
        "flag": flag,
    }


def _progress(index: int, total: int, start: float) -> None:
    elapsed = time.monotonic() - start
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    eta = f"{remaining/60:.0f}m{remaining%60:.0f}s" if remaining > 90 else f"{remaining:.0f}s"
    _safe_print(f"  [{index}/{total}] elapsed={elapsed:.0f}s ETA={eta}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _default_failure_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".failures.jsonl")


def shard_output_path(output_path: Path, shard_index: int, shard_count: int) -> Path:
    """Return *output_path* unchanged for ``shard_count == 1``, else suffixed.

    Concurrent shard processes must never append to the same output file
    (JSON lines here carry a 201-value flux array, long enough that
    concurrent appends are not guaranteed atomic), so when sharding is
    active the filename gets an explicit ``.shardIofN`` marker inserted
    before the suffix, e.g. ``snippets.jsonl`` -> ``snippets.shard0of4.jsonl``.
    This mirrors ``process_t1_kepler_batch.py``'s ``shard_output_path`` and
    is automatic (not left to the operator) so concurrent tabs can never
    accidentally collide on the same output path.
    """
    output_path = Path(output_path)
    if shard_count <= 1:
        return output_path
    return output_path.with_name(
        f"{output_path.stem}.shard{shard_index}of{shard_count}{output_path.suffix}"
    )


def build_k2_calibration_snippets(
    manifest_path: Path,
    *,
    output_path: Path,
    n_bins: int = 201,
    workers: int = 1,
    max_errors: int = 25,
    lc_fetcher: Callable[[int], tuple[list[float], list[float]] | None] | None = None,
    failure_log_path: Path | None = None,
    retry_failures: bool = False,
    commit_report: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int:
    """Fetch and fold native K2 snippets for every manifest row, with resume.

    Args:
        manifest_path: Path to ``t1_2_k2_calibration_manifest.jsonl``.
        output_path: Where to append successful snippet JSONL rows. When
            ``shard_count > 1`` this is passed through
            :func:`shard_output_path` first, so each shard writes its own file.
        n_bins: Phase bins per snippet.
        workers: Bounded thread-pool concurrency (default 1, sequential).
        max_errors: Stop early after this many consecutive non-OK results.
        lc_fetcher: Injectable per-EPIC fetcher (for tests).
        failure_log_path: Where to append terminal-failure records. Defaults
            to ``<output_path>.failures.jsonl`` (post-sharding).
        retry_failures: If True, re-attempt rows already recorded as
            terminal failures instead of skipping them.
        commit_report: If True, write and commit a run report on completion
            (set False in tests).
        shard_index: This process's shard index in ``[0, shard_count)``. Every
            concurrently running process must use the same ``shard_count``
            with a distinct ``shard_index`` -- partitioning is by
            ``epic_id % shard_count``, which is disjoint across shards, so no
            two shards ever touch the same target. Default ``0`` with
            ``shard_count=1`` (no sharding) reproduces the exact prior
            single-process behavior.
        shard_count: Total number of concurrently running shard processes.
            ``1`` (the default) disables sharding entirely.

    Returns:
        Number of snippets successfully written this run.
    """
    manifest_path = Path(manifest_path)
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    output_path = shard_output_path(Path(output_path), shard_index, shard_count)
    failure_path = failure_log_path or _default_failure_log_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if shard_count > 1:
        rows = [row for row in rows if int(row["epic_id"]) % shard_count == shard_index]

    done_keys = _completed_keys(output_path)
    failed_keys = set() if retry_failures else _terminal_failure_keys(failure_path)
    pending = [
        row for row in rows
        if _task_key(row) not in done_keys and _task_key(row) not in failed_keys
    ]

    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    total = len(pending)
    shard_note = f", shard={shard_index}/{shard_count}" if shard_count > 1 else ""
    _safe_print(
        f"T1-2 K2 native calibration snippet fetch: {total} pending "
        f"({len(rows)} in this shard's partition, {len(done_keys)} done, "
        f"{len(failed_keys)} terminal failures skipped{shard_note})  "
        f"output={output_path}"
    )

    n_written = 0
    n_failed = 0
    n_consecutive_failures = 0
    index = 0

    with (
        output_path.open("a", encoding="utf-8") as out_handle,
        failure_path.open("a", encoding="utf-8") as fail_handle,
        ThreadPoolExecutor(max_workers=max(1, workers)) as executor,
    ):
        futures = {
            executor.submit(
                build_k2_calibration_snippet, row, n_bins=n_bins, lc_fetcher=lc_fetcher
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            index += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 -- record and continue
                result = K2CalibrationSnippetResult(
                    epic_id=int(row["epic_id"]), label=int(row["label"]),
                    flux=(), period_days=float(row["period_days"]),
                    epoch_bjd=float(row["epoch_bjd"]), n_bins=n_bins,
                    flag=f"ERROR:{exc}",
                )

            if result.flag == "OK":
                out_handle.write(json.dumps(_record_from_result(result)) + "\n")
                out_handle.flush()
                n_written += 1
                n_consecutive_failures = 0
            else:
                fail_handle.write(json.dumps(_failure_record(row, result.flag)) + "\n")
                fail_handle.flush()
                n_failed += 1
                n_consecutive_failures += 1

            _progress(index, total, start)

            if n_consecutive_failures >= max_errors:
                _safe_print(
                    f"Stopping early: {n_consecutive_failures} consecutive "
                    "non-OK results."
                )
                for pending_future in futures:
                    pending_future.cancel()
                break

    completed_at = datetime.now(UTC).isoformat()
    elapsed = time.monotonic() - start
    _safe_print(
        f"Done in {elapsed:.0f}s: {n_written} written, {n_failed} failed "
        f"(of {index} attempted, {total} pending)."
    )

    if commit_report:
        report = RunReport(
            script="fetch_t1_2_k2_calibration_snippets",
            status="success" if n_written > 0 or total == 0 else "partial",
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            items_processed=index,
            items_written=n_written,
            items_failed=n_failed,
            output_paths=(str(output_path),),
            shard_index=shard_index,
            shard_count=shard_count,
            items_done_total=len(done_keys) + n_written,
            items_total=len(rows),
            percent_done=(
                round(100.0 * (len(done_keys) + n_written) / len(rows), 2)
                if rows
                else None
            ),
        )
        report_path = report_path_for(
            "fetch_t1_2_k2_calibration_snippets",
            shard_index=shard_index,
            shard_count=shard_count,
        )
        run_and_commit_report(report, report_path)

    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch native K2 light curves for the T1-2 held-out calibration "
            "manifest and write phase-folded, normalised snippets."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_manifest.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/t1_2_k2_calibration_snippets.jsonl")
    )
    parser.add_argument("--n-bins", type=int, default=201)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-errors", type=int, default=25)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "This process's shard index in [0, --shard-count), for running many "
            "console tabs concurrently against disjoint target sets. All "
            "concurrently running tabs must share --shard-count, each with its "
            "own unique --shard-index (default: 0)."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Total number of concurrently running shard processes. 1 (default) "
            "disables sharding. Partitioning is epic_id %% shard_count, so "
            "shards never touch the same target; each shard's output file is "
            "auto-suffixed (see shard_output_path)."
        ),
    )
    args = parser.parse_args(argv)

    build_k2_calibration_snippets(
        args.manifest,
        output_path=args.output,
        n_bins=args.n_bins,
        workers=args.workers,
        max_errors=args.max_errors,
        retry_failures=args.retry_failures,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
