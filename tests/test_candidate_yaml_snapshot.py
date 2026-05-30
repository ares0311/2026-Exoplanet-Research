"""Tests for Skills/candidate_yaml_snapshot.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_yaml_snapshot import build_yaml_snapshot, format_yaml_preview, write_yaml_snapshot


class TestBuildYamlSnapshot:
    def _candidate(self) -> dict[str, object]:
        return {"tic_id": "12345", "period_days": 5.0, "fpp": 0.1, "snr": 15.0}

    def test_basic_snapshot(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        assert r.flag == "OK"
        assert r.n_fields > 0

    def test_tic_id_extracted(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        assert r.tic_id == "12345"

    def test_yaml_contains_period(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        assert "period_days" in r.yaml_text

    def test_yaml_starts_with_triple_dash(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        assert r.yaml_text.startswith("---")

    def test_empty_candidate_no_fields(self) -> None:
        r = build_yaml_snapshot({})
        assert r.flag == "NO_FIELDS"
        assert r.n_fields == 0

    def test_custom_fields(self) -> None:
        r = build_yaml_snapshot({"tic_id": "99", "fpp": 0.2}, fields=["tic_id", "fpp"])
        assert r.n_fields == 2

    def test_result_frozen(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_write_creates_file(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.yaml"
            write_yaml_snapshot(r, path)
            assert path.exists()

    def test_written_file_contains_yaml(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.yaml"
            write_yaml_snapshot(r, path)
            content = path.read_text()
            assert "---" in content

    def test_format_returns_string(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        s = format_yaml_preview(r)
        assert isinstance(s, str)
        assert "YAML" in s

    def test_format_contains_tic_id(self) -> None:
        r = build_yaml_snapshot(self._candidate())
        s = format_yaml_preview(r)
        assert "12345" in s

    def test_unknown_tic_id_fallback(self) -> None:
        r = build_yaml_snapshot({"period_days": 5.0})
        assert r.tic_id == "unknown"
