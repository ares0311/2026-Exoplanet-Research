"""Tests for Skills.fetch_additional_tess_labels."""
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

    def test_uses_current_venv_safe_v3_expansion_recipe(self) -> None:
        rows = [{"tic_id": 1, "label": 1, "source": "exofop_toi"}]
        result = format_expansion_summary({100, 200}, rows)
        assert "git pull origin main" in result
        assert "caffeinate -dims .venv/bin/python Skills/fetch_tess_lc_snippets.py" in result
        assert "caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py" in result
        assert "data/tess_snippets_v2.jsonl" in result
        assert "data/tess_snippets_expansion_v3.jsonl" in result
        assert "data/tess_snippets_v3.jsonl" in result
        assert "data/tess_cnn_splits_v3" in result
        assert "caffeinate -dims python" not in result
        assert "data/cnn_splits_v2" not in result


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_notes(self) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_additional_tess_labels import _write_run_report

        with patch(
            "Skills.fetch_additional_tess_labels.run_and_commit_report",
            return_value=True,
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                stats={
                    "total_labeled": 10, "new_tic_ids": 3, "fetch_failures": 0,
                    "corpus_size": 100, "positive_only": 0,
                },
                output_path=Path("data/new_tess_targets.txt"),
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "fetch_additional_tess_labels"
        assert report.status == "success"
        assert report.items_processed == 10
        assert report.items_written == 3
        assert report.items_failed == 0
        assert "corpus_size=100" in report.notes
        assert "positive_only=0" in report.notes
        assert path.name == "fetch_additional_tess_labels.jsonl"

    def test_git_run_fn_is_threaded_through(self) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_additional_tess_labels import _write_run_report

        fake_runner = MagicMock()
        with patch(
            "Skills.fetch_additional_tess_labels.run_and_commit_report",
            return_value=True,
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.txt"),
                git_run_fn=fake_runner,
            )
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_commit_failure_warns_but_does_not_raise(self, capsys) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_additional_tess_labels import _write_run_report

        with patch(
            "Skills.fetch_additional_tess_labels.run_and_commit_report",
            return_value=False,
        ):
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.txt"),
                git_run_fn=MagicMock(),
            )
        assert "Warning" in capsys.readouterr().out

    def test_cli_writes_run_report_with_injected_git_runner(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_additional_tess_labels import _cli

        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("")
        out = tmp_path / "targets.txt"
        fake_runner = MagicMock()

        with (
            patch(
                "Skills.fetch_additional_tess_labels.fetch_toi_labels",
                return_value=[
                    {"tic_id": 1, "label": 1, "disposition": "CP",
                     "period_days": 1.0, "epoch_bjd": 0.0, "source": "exofop_toi"},
                ],
            ),
            patch(
                "Skills.fetch_additional_tess_labels.fetch_ctoi_labels",
                return_value=[],
            ),
            patch(
                "Skills.fetch_additional_tess_labels.run_and_commit_report",
                return_value=True,
            ) as commit,
        ):
            exit_code = _cli(
                ["--corpus", str(corpus), "--output", str(out)],
                git_run_fn=fake_runner,
            )

        assert exit_code == 0
        report, _path = commit.call_args.args
        assert report.items_processed == 1
        assert report.items_written == 1
        assert report.items_failed == 0
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_cli_counts_fetch_failures_but_still_succeeds(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from Skills.fetch_additional_tess_labels import _cli

        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("")
        out = tmp_path / "targets.txt"

        with (
            patch(
                "Skills.fetch_additional_tess_labels.fetch_toi_labels",
                side_effect=ConnectionError("no network"),
            ),
            patch(
                "Skills.fetch_additional_tess_labels.fetch_ctoi_labels",
                side_effect=ConnectionError("no network"),
            ),
            patch(
                "Skills.fetch_additional_tess_labels.run_and_commit_report",
                return_value=True,
            ) as commit,
        ):
            exit_code = _cli(["--corpus", str(corpus), "--output", str(out)])

        assert exit_code == 0
        report, _path = commit.call_args.args
        assert report.items_failed == 2
        assert report.items_processed == 0
        assert report.items_written == 0
