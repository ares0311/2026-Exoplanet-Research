"""Tests for Skills/fetch_confirmed_hosts.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.fetch_confirmed_hosts import (  # noqa: E402
    _NEA_TAP_URL,
    _QUERY,
    fetch_confirmed_host_tic_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(*rows: tuple[str, ...], header: str = "tic_id") -> str:
    lines = [header] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_frozenset(self) -> None:
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: _make_csv(("111",)))
        assert isinstance(result, frozenset)

    def test_elements_are_ints(self) -> None:
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: _make_csv(("42",)))
        assert all(isinstance(v, int) for v in result)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_single_row(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("12345",))
        )
        assert 12345 in result

    def test_multiple_rows(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("10",), ("20",), ("30",))
        )
        assert result == frozenset({10, 20, 30})

    def test_float_encoded_id(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("99999.0",))
        )
        assert 99999 in result

    def test_skips_empty_values(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("",), ("55",))
        )
        assert result == frozenset({55})

    def test_skips_whitespace_only(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("  ",), ("77",))
        )
        assert result == frozenset({77})

    def test_skips_non_numeric(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("abc",), ("88",))
        )
        assert result == frozenset({88})

    def test_deduplicates(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("100",), ("100",))
        )
        assert result == frozenset({100})

    def test_empty_csv_body(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: "tic_id\n"
        )
        assert result == frozenset()


# ---------------------------------------------------------------------------
# Failure modes (fail-open)
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_network_error_returns_empty(self) -> None:
        def bad_fetch(_: str) -> str:
            raise OSError("connection refused")

        result = fetch_confirmed_host_tic_ids(fetch_fn=bad_fetch)
        assert result == frozenset()

    def test_malformed_csv_returns_empty(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: "\x00\x01\x02"
        )
        assert isinstance(result, frozenset)

    def test_missing_tic_id_column_returns_empty(self) -> None:
        csv = "other_col\n1\n2\n"
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: csv)
        assert result == frozenset()


# ---------------------------------------------------------------------------
# URL and query
# ---------------------------------------------------------------------------


class TestUrl:
    def test_url_contains_tap_endpoint(self) -> None:
        assert "exoplanetarchive.ipac.caltech.edu" in _NEA_TAP_URL
        assert "TAP" in _NEA_TAP_URL

    def test_query_targets_ps_table(self) -> None:
        assert "FROM ps" in _QUERY

    def test_query_filters_transiting(self) -> None:
        assert "pl_tranflag=1" in _QUERY

    def test_query_filters_default_flag(self) -> None:
        assert "default_flag=1" in _QUERY

    def test_fetch_fn_receives_url(self) -> None:
        received: list[str] = []

        def capture_fn(url: str) -> str:
            received.append(url)
            return "tic_id\n"

        fetch_confirmed_host_tic_ids(fetch_fn=capture_fn)
        assert len(received) == 1
        assert "exoplanetarchive" in received[0]
        assert "tic_id" in received[0]
