"""Tests for Skills/watchlist.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.watchlist import Watchlist  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wl(tmp_path: Path) -> Watchlist:
    return Watchlist(tmp_path / "watchlist.json")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWatchlistAdd:
    def test_add_single_id(self, wl: Watchlist) -> None:
        wl.add(12345)
        assert wl.contains(12345)

    def test_add_persists_across_reload(self, tmp_path: Path) -> None:
        w1 = Watchlist(tmp_path / "wl.json")
        w1.add(99999)
        w2 = Watchlist(tmp_path / "wl.json")
        assert w2.contains(99999)

    def test_add_with_note(self, wl: Watchlist) -> None:
        wl.add(11111, note="hot Jupiter candidate")
        entry = wl.entries()[0]
        assert entry["note"] == "hot Jupiter candidate"

    def test_add_updates_note_if_present(self, wl: Watchlist) -> None:
        wl.add(11111, note="first note")
        wl.add(11111, note="updated note")
        assert len(wl.list_ids()) == 1
        assert wl.entries()[0]["note"] == "updated note"


class TestWatchlistRemove:
    def test_remove_existing(self, wl: Watchlist) -> None:
        wl.add(22222)
        result = wl.remove(22222)
        assert result is True
        assert not wl.contains(22222)

    def test_remove_missing_returns_false(self, wl: Watchlist) -> None:
        assert wl.remove(99999) is False

    def test_remove_persists(self, tmp_path: Path) -> None:
        w1 = Watchlist(tmp_path / "wl.json")
        w1.add(33333)
        w1.remove(33333)
        w2 = Watchlist(tmp_path / "wl.json")
        assert not w2.contains(33333)


class TestWatchlistQuery:
    def test_list_ids_sorted(self, wl: Watchlist) -> None:
        for tid in [500, 100, 300]:
            wl.add(tid)
        assert wl.list_ids() == [100, 300, 500]

    def test_contains_false_for_absent(self, wl: Watchlist) -> None:
        assert not wl.contains(0)

    def test_entries_have_required_keys(self, wl: Watchlist) -> None:
        wl.add(77777, note="test")
        e = wl.entries()[0]
        for key in ("tic_id", "note", "added_at"):
            assert key in e

    def test_summary_returns_count(self, wl: Watchlist) -> None:
        wl.add(1)
        wl.add(2)
        s = wl.summary()
        assert s["n_entries"] == 2

    def test_clear_empties_list(self, wl: Watchlist) -> None:
        wl.add(1)
        wl.add(2)
        wl.clear()
        assert wl.list_ids() == []

    def test_new_watchlist_starts_empty(self, wl: Watchlist) -> None:
        assert wl.list_ids() == []
