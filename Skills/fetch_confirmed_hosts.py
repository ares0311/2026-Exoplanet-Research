"""Fetch TIC IDs of confirmed transiting planet hosts from NASA Exoplanet Archive.

Queries the ``ps`` (planetary systems) TAP table for rows where
``tran_flag=1`` (transiting geometry confirmed) and ``tic_id IS NOT NULL``.

Public API
----------
fetch_confirmed_host_tic_ids(*, fetch_fn=None, strict=False) -> frozenset[int]
    Return a frozenset of integer TIC IDs to exclude from discovery scans.
    On any network or parse failure returns an empty frozenset (fails open,
    so the scan continues without confirmed-planet exclusion rather than
    crashing). Strict mode raises for immutable metadata preparation.
"""
from __future__ import annotations

import csv
import io
import ssl
import urllib.parse
import urllib.request
from collections.abc import Callable

_NEA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

_QUERY = (
    "SELECT DISTINCT tic_id FROM ps "
    "WHERE tran_flag=1 AND default_flag=1 AND tic_id IS NOT NULL"
)


def _default_fetch(url: str) -> str:
    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def fetch_confirmed_host_tic_ids(
    *,
    fetch_fn: Callable[[str], str] | None = None,
    strict: bool = False,
) -> frozenset[int]:
    """Return TIC IDs of confirmed transiting planet hosts.

    Args:
        fetch_fn: Injectable HTTP fetch callable (accepts URL, returns CSV
            string).  Defaults to ``urllib.request.urlopen``.

    Returns:
        Frozenset of integer TIC IDs. Empty frozenset on any failure unless
        ``strict=True``, which raises so production manifest preparation fails closed.
    """
    _fetch = fetch_fn or _default_fetch

    params = urllib.parse.urlencode({"query": _QUERY, "format": "csv"})
    url = f"{_NEA_TAP_URL}?{params}"

    try:
        raw = _fetch(url)
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()

    try:
        reader = csv.DictReader(io.StringIO(raw))
        tic_ids: set[int] = set()
        for row in reader:
            val = (row.get("tic_id") or "").strip()
            if not val:
                continue
            if val.upper().startswith("TIC "):
                val = val[4:].strip()
            try:
                tic_ids.add(int(float(val)))
            except (ValueError, TypeError):
                continue
        result = frozenset(tic_ids)
        if strict and not result:
            raise RuntimeError("Confirmed-host TAP query returned no TIC IDs")
        return result
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return frozenset()
