"""Tests for Skills/fetch_tess_toi.py (offline / unit tests only)."""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.fetch_tess_toi import _COL_MAP, _KEEP_DISPOSITIONS


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV without pandas (avoids monkeypatch interference)."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _mock_toi_df() -> pd.DataFrame:
    rows = [
        {"TOI": 700.01, "TIC ID": 150428135, "TFOPWG Disposition": "CP",
         "Period (days)": 37.4, "Epoch (BJD)": 2458325.0,
         "Duration (hours)": 2.3, "Depth (mmag)": 1.2,
         "Planet Radius (R_Earth)": 1.8, "Planet SNR": 22.0,
         "Number of Sectors": 6, "Stellar Radius (R_Sun)": 0.42,
         "Stellar Eff Temp (K)": 3480, "Stellar log(g) (cm/s^2)": 4.8,
         "TESS Mag": 13.1},
        {"TOI": 101.01, "TIC ID": 777, "TFOPWG Disposition": "KP",
         "Period (days)": 10.0, "Epoch (BJD)": 2458320.0,
         "Duration (hours)": 2.0, "Depth (mmag)": 2.0,
         "Planet Radius (R_Earth)": 1.5, "Planet SNR": 30.0,
         "Number of Sectors": 4, "Stellar Radius (R_Sun)": 0.9,
         "Stellar Eff Temp (K)": 5500, "Stellar log(g) (cm/s^2)": 4.4,
         "TESS Mag": 11.0},
        {"TOI": 100.01, "TIC ID": 999, "TFOPWG Disposition": "FP",
         "Period (days)": 3.1, "Epoch (BJD)": 2458310.0,
         "Duration (hours)": 1.0, "Depth (mmag)": 20.0,
         "Planet Radius (R_Earth)": 22.0, "Planet SNR": 5.0,
         "Number of Sectors": 2, "Stellar Radius (R_Sun)": 1.0,
         "Stellar Eff Temp (K)": 5800, "Stellar log(g) (cm/s^2)": 4.4,
         "TESS Mag": 10.5},
        {"TOI": 102.01, "TIC ID": 666, "TFOPWG Disposition": "FA",
         "Period (days)": 1.0, "Epoch (BJD)": 2458300.0,
         "Duration (hours)": 0.5, "Depth (mmag)": 0.1,
         "Planet Radius (R_Earth)": 0.5, "Planet SNR": 2.0,
         "Number of Sectors": 1, "Stellar Radius (R_Sun)": 1.1,
         "Stellar Eff Temp (K)": 6000, "Stellar log(g) (cm/s^2)": 4.3,
         "TESS Mag": 9.0},
        {"TOI": 200.01, "TIC ID": 888, "TFOPWG Disposition": "PC",
         "Period (days)": 5.0, "Epoch (BJD)": 2458290.0,
         "Duration (hours)": 1.5, "Depth (mmag)": 3.0,
         "Planet Radius (R_Earth)": 2.5, "Planet SNR": 15.0,
         "Number of Sectors": 3, "Stellar Radius (R_Sun)": 0.9,
         "Stellar Eff Temp (K)": 5200, "Stellar log(g) (cm/s^2)": 4.5,
         "TESS Mag": 11.0},
    ]
    return pd.DataFrame(rows)


def _mock_fetch(url: str) -> bytes:
    """Return mock TOI table as CSV bytes without hitting the network."""
    buf = io.StringIO()
    _mock_toi_df().to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_col_map_has_toi(self) -> None:
        assert "TOI" in _COL_MAP

    def test_col_map_has_epoch(self) -> None:
        assert "Epoch (BJD)" in _COL_MAP

    def test_col_map_has_disposition(self) -> None:
        assert "TFOPWG Disposition" in _COL_MAP

    def test_keep_dispositions_positive_class(self) -> None:
        assert "CP" in _KEEP_DISPOSITIONS
        assert "KP" in _KEEP_DISPOSITIONS

    def test_keep_dispositions_negative_class(self) -> None:
        assert "FP" in _KEEP_DISPOSITIONS
        assert "FA" in _KEEP_DISPOSITIONS

    def test_eb_not_kept(self) -> None:
        assert "EB" not in _KEEP_DISPOSITIONS

    def test_pc_not_kept(self) -> None:
        assert "PC" not in _KEEP_DISPOSITIONS

    def test_normalised_names_unique(self) -> None:
        assert len(set(_COL_MAP.values())) == len(_COL_MAP)


# ---------------------------------------------------------------------------
# fetch_toi_table — offline via injectable fetch_fn
# ---------------------------------------------------------------------------


class TestFetchToiTable:
    def test_saves_csv(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        result = fetch_toi_table(out, fetch_fn=_mock_fetch)
        assert result == out
        assert out.exists()

    def test_filters_to_kept_dispositions(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        dispositions = {r["tfopwg_disposition"] for r in rows}
        assert dispositions <= {"CP", "KP", "FP", "FA"}

    def test_pc_excluded(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert all(r["tfopwg_disposition"] != "PC" for r in rows)

    def test_kp_included(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert any(r["tfopwg_disposition"] == "KP" for r in rows)

    def test_fa_included(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert any(r["tfopwg_disposition"] == "FA" for r in rows)

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "subdir" / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        assert out.exists()

    def test_columns_normalised(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert rows
        assert "toi" in rows[0]
        assert "tfopwg_disposition" in rows[0]

    def test_snr_column_present(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert rows
        assert "snr" in rows[0]

    def test_epoch_bjd_column_present(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        fetch_toi_table(out, fetch_fn=_mock_fetch)
        rows = _read_csv(out)
        assert rows
        assert "epoch_bjd" in rows[0]

    def test_stats_populated(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        stats: dict[str, int] = {}
        fetch_toi_table(out, fetch_fn=_mock_fetch, stats=stats)
        # 5 mock rows, PC excluded by disposition -> 4 kept, all valid ephemerides.
        assert stats["written"] == 4
        assert stats["errors"] == 0
        assert stats["total"] == 4
        assert stats["rejected_ephemerides"] == 0

    def test_stats_unset_when_not_requested(self, tmp_path: Path) -> None:
        from Skills.fetch_tess_toi import fetch_toi_table
        out = tmp_path / "toi.csv"
        result = fetch_toi_table(out, fetch_fn=_mock_fetch)
        assert result.exists()


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_notes(self) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_tess_toi import _write_run_report

        with patch(
            "Skills.fetch_tess_toi.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                stats={"written": 4, "errors": 0, "total": 4, "rejected_ephemerides": 1},
                output_path=Path("data/tess_toi.csv"),
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "fetch_tess_toi"
        assert report.status == "success"
        assert report.items_processed == 4
        assert report.items_written == 4
        assert report.items_failed == 0
        assert "rejected_ephemerides=1" in report.notes
        assert path.name == "fetch_tess_toi.jsonl"

    def test_git_run_fn_is_threaded_through(self) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_tess_toi import _write_run_report

        fake_runner = MagicMock()
        with patch(
            "Skills.fetch_tess_toi.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.csv"),
                git_run_fn=fake_runner,
            )
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_commit_failure_warns_but_does_not_raise(self, capsys) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_tess_toi import _write_run_report

        with patch(
            "Skills.fetch_tess_toi.run_and_commit_report", return_value=False
        ):
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.csv"),
                git_run_fn=MagicMock(),
            )
        assert "Warning" in capsys.readouterr().out

    def test_cli_writes_run_report_with_injected_git_runner(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_tess_toi import _cli

        out = tmp_path / "toi.csv"
        fake_runner = MagicMock()

        def fake_fetch(output_path, *, stats=None):
            if stats is not None:
                stats["written"] = 4
                stats["errors"] = 0
                stats["total"] = 4
                stats["rejected_ephemerides"] = 0
            return Path(output_path)

        with (
            patch("Skills.fetch_tess_toi.fetch_toi_table", side_effect=fake_fetch),
            patch(
                "Skills.fetch_tess_toi.run_and_commit_report", return_value=True
            ) as commit,
        ):
            exit_code = _cli(["--output", str(out)], git_run_fn=fake_runner)

        assert exit_code == 0
        commit.assert_called_once()
        assert commit.call_args.kwargs["run_fn"] is fake_runner
        report, _path = commit.call_args.args
        assert report.items_written == 4
