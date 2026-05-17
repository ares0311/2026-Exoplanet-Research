"""Tests for Skills.archive_lookup."""
from __future__ import annotations

from Skills.archive_lookup import ArchiveStatus, check_archive, format_archive_status


def _nop(tic_id: int) -> list[dict]:
    return []


class TestCheckArchive:
    def test_returns_archive_status(self) -> None:
        status = check_archive(1, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        assert isinstance(status, ArchiveStatus)

    def test_tic_id_stored(self) -> None:
        status = check_archive(99999, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        assert status.tic_id == 99999

    def test_novel_when_no_hits(self) -> None:
        status = check_archive(1, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        assert status.recommendation == "novel"
        assert not status.in_nea
        assert not status.in_toi

    def test_known_object_when_in_nea(self) -> None:
        status = check_archive(
            1,
            nea_fn=lambda t: [{"pl_name": "TOI-700 d"}],
            toi_fn=_nop,
            koi_fn=_nop,
        )
        assert status.in_nea
        assert status.recommendation == "known_object"
        assert "TOI-700 d" in status.nea_planets

    def test_toi_followup_when_in_toi_not_nea(self) -> None:
        status = check_archive(
            1,
            nea_fn=_nop,
            toi_fn=lambda t: [{"TOI": "700.01"}],
            koi_fn=_nop,
        )
        assert status.in_toi
        assert status.recommendation == "toi_followup"

    def test_koi_entries_stored(self) -> None:
        status = check_archive(
            1,
            nea_fn=_nop,
            toi_fn=_nop,
            koi_fn=lambda t: [{"kepoi_name": "K00001.01"}],
        )
        assert "K00001.01" in status.koi_entries

    def test_exception_in_nea_does_not_crash(self) -> None:
        def _bad(t: int) -> list[dict]:
            raise RuntimeError("network error")
        status = check_archive(1, nea_fn=_bad, toi_fn=_nop, koi_fn=_nop)
        assert status.recommendation == "novel"

    def test_exception_in_toi_does_not_crash(self) -> None:
        def _bad(t: int) -> list[dict]:
            raise RuntimeError("network error")
        status = check_archive(1, nea_fn=_nop, toi_fn=_bad, koi_fn=_nop)
        assert isinstance(status, ArchiveStatus)

    def test_nea_takes_precedence_over_toi(self) -> None:
        status = check_archive(
            1,
            nea_fn=lambda t: [{"pl_name": "My Planet b"}],
            toi_fn=lambda t: [{"TOI": "1.01"}],
            koi_fn=_nop,
        )
        assert status.recommendation == "known_object"

    def test_found_in_empty_tuple_when_no_hits(self) -> None:
        status = check_archive(1, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        assert status.nea_planets == ()
        assert status.toi_entries == ()


class TestFormatArchiveStatus:
    def test_format_contains_tic_id(self) -> None:
        status = check_archive(42, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        text = format_archive_status(status)
        assert "42" in text

    def test_format_contains_recommendation(self) -> None:
        status = check_archive(1, nea_fn=_nop, toi_fn=_nop, koi_fn=_nop)
        text = format_archive_status(status)
        assert "novel" in text.lower()
