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
                      resume, max_errors) -> int
"""
from __future__ import annotations

import contextlib
import json
import socket
import time
from collections.abc import Callable
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
) -> int:
    """Build phase-folded CNN snippets for a list of KOI rows.

    Writes one JSONL record per successful snippet.  Skips already-written
    kepids when ``resume=True``.

    Args:
        koi_rows: List of dicts (from :func:`fetch_koi_table`).
        n_bins: Number of phase bins per snippet.
        output_path: Destination JSONL file path.
        lc_fetcher: Injectable light-curve fetcher.
        resume: Skip kepids already present in output_path.
        max_errors: Abort after this many consecutive errors.

    Returns:
        Number of snippets written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[int] = set()
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        already_done.add(int(row.get("kepid", -1)))
                    except json.JSONDecodeError:
                        pass

    n_total = len(koi_rows)
    n_written = 0
    n_errors = 0
    consecutive_errors = 0
    start = time.monotonic()

    print(
        f"Building Kepler snippets: {n_total} KOIs  resume={resume}  "
        f"already_done={len(already_done)}  output={output_path}",
        flush=True,
    )

    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as fh:
        for i, row in enumerate(koi_rows, 1):
            kepid = int(row.get("kepid", 0))
            disposition = str(row.get("koi_disposition", ""))
            period = float(row.get("koi_period", 0.0))
            time0bk = float(row.get("koi_time0bk", 0.0))
            epoch_bjd = time0bk + _KEPLER_BJD_OFFSET

            if kepid in already_done:
                continue
            if period <= 0 or epoch_bjd <= _KEPLER_BJD_OFFSET:
                continue

            label = 1 if disposition == "CONFIRMED" else 0

            result = build_kepler_snippet(
                kepid, label, period, epoch_bjd,
                n_bins=n_bins, lc_fetcher=lc_fetcher,
            )

            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 1.0
            remaining = (n_total - i) / rate
            eta = (
                f"{remaining/60:.0f}m{remaining%60:.0f}s"
                if remaining > 90
                else f"{remaining:.0f}s"
            )

            if result.flag == "OK":
                record = {
                    "tic_id": 0,
                    "kepid": kepid,
                    "label": label,
                    "flux": list(result.flux),
                    "source": "kepler",
                    "period_days": period,
                    "epoch_bjd": epoch_bjd,
                    "n_bins": n_bins,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                n_written += 1
                consecutive_errors = 0
                print(
                    f"  [{i}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                    f" KIC {kepid} label={label}"
                    f"  written={n_written}  elapsed={elapsed:.0f}s  ETA={eta}",
                    flush=True,
                )
            else:
                n_errors += 1
                consecutive_errors += 1
                print(
                    f"  [{i}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                    f" KIC {kepid} SKIP flag={result.flag}"
                    f"  errors={n_errors}  elapsed={elapsed:.0f}s  ETA={eta}",
                    flush=True,
                )
                if consecutive_errors >= max_errors:
                    print(
                        f"Aborting: {consecutive_errors} consecutive errors.",
                        flush=True,
                    )
                    break

    print(
        f"Done. {n_written} snippets written, {n_errors} skipped/errors.",
        flush=True,
    )
    return n_written


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
    )
    print(f"Flag: OK  snippets_written={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
