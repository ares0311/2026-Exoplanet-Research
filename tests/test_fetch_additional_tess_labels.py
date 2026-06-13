"""Tests for Skills.fetch_additional_tess_labels (13 tests)."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.fetch_additional_tess_labels import (
    fetch_ctoi_labels,
    fetch_toi_labels,
    find_new_tic_ids,
    format_expansion_summary,
    load_corpus_tic_ids,
    write_target_list,
)

# ---------------------------------------------------------------------------
# load_corpus_tic_ids
# ---------------------------------------------------------------------------


class TestLoadCorpusTicIds:
    def test_empty_file_returns_empty_set(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = load_corpus_tic_ids(p)
        assert result == set()

    def test_missing_file_returns_empty_set(self, tmp_path: Path) -> None:
        result = load_corpus_tic_ids(tmp_path / "nonexistent.jsonl")
        assert result == set()

    def test_extracts_tic_ids(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"tic_id": 123, "flux": [], "label": 1}),
            json.dumps({"tic_id": 456, "flux": [], "label": 0}),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = load_corpus_tic_ids(p)
        assert result == {123, 456}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text('{"tic_id": 999}\nnot_json\n{"tic_id": 1}\n')
        result = load_corpus_tic_ids(p)
        assert 999 in result
        assert 1 in result


# ---------------------------------------------------------------------------
# fetch_toi_labels / fetch_ctoi_labels
# ---------------------------------------------------------------------------


def _make_toi_fn(rows: list[dict]):
    def fn():
        return rows
    return fn


class TestFetchTOILabels:
    def test_confirmed_planet_yields_label_1(self) -> None:
        rows = [{"TIC ID": "150428135", "TFOPWG Disposition": "CP",
                 "Period (days)": "9.9", "Epoch (BJD)": "2458325.5"}]
        result = fetch_toi_labels(_make_toi_fn(rows))
        assert len(result) == 1
        assert result[0]["label"] == 1
        assert result[0]["tic_id"] == 150428135

    def test_false_positive_yields_label_0(self) -> None:
        rows = [{"TIC ID": "12345", "TFOPWG Disposition": "FP",
                 "Period (days)": "5.0", "Epoch (BJD)": "2458300.0"}]
        result = fetch_toi_labels(_make_toi_fn(rows))
        assert result[0]["label"] == 0

    def test_unknown_disposition_excluded(self) -> None:
        rows = [{"TIC ID": "99999", "TFOPWG Disposition": "PC",
                 "Period (days)": "1.0", "Epoch (BJD)": "2458000.0"}]
        result = fetch_toi_labels(_make_toi_fn(rows))
        assert len(result) == 0

    def test_known_planet_yields_label_1(self) -> None:
        rows = [{"TIC ID": "200", "TFOPWG Disposition": "KP",
                 "Period (days)": "3.0", "Epoch (BJD)": "2458100.0"}]
        result = fetch_toi_labels(_make_toi_fn(rows))
        assert result[0]["label"] == 1

    def test_source_field(self) -> None:
        rows = [{"TIC ID": "111", "TFOPWG Disposition": "CP",
                 "Period (days)": "2.0", "Epoch (BJD)": "2458000.0"}]
        result = fetch_toi_labels(_make_toi_fn(rows))
        assert result[0]["source"] == "exofop_toi"


class TestFetchCTOILabels:
    def test_ctoi_source_field(self) -> None:
        rows = [{"TIC ID": "333", "User Disposition": "FP",
                 "Period (days)": "1.5", "Epoch (BJD)": "2458000.0"}]
        result = fetch_ctoi_labels(_make_toi_fn(rows))
        assert result[0]["source"] == "exofop_ctoi"


# ---------------------------------------------------------------------------
# find_new_tic_ids
# ---------------------------------------------------------------------------


class TestFindNewTicIds:
    def _make_rows(self) -> list[dict]:
        return [
            {"tic_id": 100, "label": 1, "disposition": "CP", "source": "toi",
             "period_days": 3.0, "epoch_bjd": 2458000.0},
            {"tic_id": 200, "label": 0, "disposition": "FP", "source": "toi",
             "period_days": 5.0, "epoch_bjd": 2458001.0},
            {"tic_id": 300, "label": 1, "disposition": "CP", "source": "ctoi",
             "period_days": 2.0, "epoch_bjd": 2458002.0},
        ]

    def test_excludes_existing_corpus_ids(self) -> None:
        rows = self._make_rows()
        new = find_new_tic_ids({100, 200}, rows)
        assert [r["tic_id"] for r in new] == [300]

    def test_positive_only_filter(self) -> None:
        rows = self._make_rows()
        new = find_new_tic_ids(set(), rows, positive_only=True)
        assert all(r["label"] == 1 for r in new)
        assert len(new) == 2

    def test_deduplication(self) -> None:
        rows = self._make_rows() + self._make_rows()
        new = find_new_tic_ids(set(), rows)
        tic_ids = [r["tic_id"] for r in new]
        assert len(tic_ids) == len(set(tic_ids))


# ---------------------------------------------------------------------------
# write_target_list / format_expansion_summary
# ---------------------------------------------------------------------------


class TestWriteTargetList:
    def test_writes_tic_ids_one_per_line(self, tmp_path: Path) -> None:
        rows = [
            {"tic_id": 111, "label": 1, "source": "toi"},
            {"tic_id": 222, "label": 0, "source": "toi"},
        ]
        out = tmp_path / "targets.txt"
        n = write_target_list(rows, out)
        assert n == 2
        lines = [ln.strip() for ln in out.read_text().splitlines() if ln.strip()]
        assert lines == ["111", "222"]

    def test_companion_json_written(self, tmp_path: Path) -> None:
        rows = [{"tic_id": 555, "label": 1, "source": "toi"}]
        out = tmp_path / "targets.txt"
        write_target_list(rows, out)
        meta = json.loads((tmp_path / "targets.json").read_text())
        assert meta[0]["tic_id"] == 555


class TestFormatExpansionSummary:
    def test_returns_string(self) -> None:
        rows = [{"tic_id": 1, "label": 1, "source": "exofop_toi"}]
        result = format_expansion_summary({100, 200}, rows)
        assert isinstance(result, str)
        assert "Existing corpus TIC IDs" in result
        assert "New labeled TIC IDs found: 1" in result
