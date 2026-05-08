"""Tests for Skills/fetch_tess_toi.py (offline / unit tests only)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.fetch_tess_toi import _COL_MAP, _KEEP_DISPOSITIONS


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV without pandas (avoids monkeypatch interference)."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_col_map_has_toi(self) -> None:
        assert "TOI" in _COL_MAP

    def test_col_map_has_disposition(self) -> None:
        assert "TFOPWG Disposition" in _COL_MAP

    def test_keep_dispositions(self) -> None:
        assert "CP" in _KEEP_DISPOSITIONS
        assert "FP" in _KEEP_DISPOSITIONS
        assert "EB" in _KEEP_DISPOSITIONS

    def test_pc_not_kept(self) -> None:
        assert "PC" not in _KEEP_DISPOSITIONS

    def test_normalised_names_unique(self) -> None:
        assert len(set(_COL_MAP.values())) == len(_COL_MAP)


# ---------------------------------------------------------------------------
# fetch_toi_table — offline via monkeypatch
# ---------------------------------------------------------------------------


def _mock_toi_df() -> pd.DataFrame:
    rows = [
        {"TOI": 700.01, "TIC ID": 150428135, "TFOPWG Disposition": "CP",
         "Period (days)": 37.4, "Duration (hours)": 2.3, "Depth (mmag)": 1.2,
         "Planet Radius (R_Earth)": 1.8, "Planet SNR": 22.0,
         "Number of Sectors": 6, "Stellar Radius (R_Sun)": 0.42,
         "Stellar Eff Temp (K)": 3480, "Stellar log(g) (cm/s^2)": 4.8,
         "TESS Mag": 13.1},
        {"TOI": 100.01, "TIC ID": 999, "TFOPWG Disposition": "FP",
         "Period (days)": 3.1, "Duration (hours)": 1.0, "Depth (mmag)": 20.0,
         "Planet Radius (R_Earth)": 22.0, "Planet SNR": 5.0,
         "Number of Sectors": 2, "Stellar Radius (R_Sun)": 1.0,
         "Stellar Eff Temp (K)": 5800, "Stellar log(g) (cm/s^2)": 4.4,
         "TESS Mag": 10.5},
        {"TOI": 200.01, "TIC ID": 888, "TFOPWG Disposition": "PC",
         "Period (days)": 5.0, "Duration (hours)": 1.5, "Depth (mmag)": 3.0,
         "Planet Radius (R_Earth)": 2.5, "Planet SNR": 15.0,
         "Number of Sectors": 3, "Stellar Radius (R_Sun)": 0.9,
         "Stellar Eff Temp (K)": 5200, "Stellar log(g) (cm/s^2)": 4.5,
         "TESS Mag": 11.0},
    ]
    return pd.DataFrame(rows)


class TestFetchToiTable:
    def test_saves_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr(
            "pandas.read_csv", lambda *a, **kw: _mock_toi_df()
        )
        out = tmp_path / "toi.csv"
        result = mod.fetch_toi_table(out)
        assert result == out
        assert out.exists()

    def test_filters_to_kept_dispositions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: _mock_toi_df())
        out = tmp_path / "toi.csv"
        mod.fetch_toi_table(out)
        rows = _read_csv(out)
        dispositions = {r["tfopwg_disposition"] for r in rows}
        assert dispositions <= {"CP", "FP", "EB"}

    def test_pc_excluded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: _mock_toi_df())
        out = tmp_path / "toi.csv"
        mod.fetch_toi_table(out)
        rows = _read_csv(out)
        assert all(r["tfopwg_disposition"] != "PC" for r in rows)

    def test_creates_parent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: _mock_toi_df())
        out = tmp_path / "subdir" / "toi.csv"
        mod.fetch_toi_table(out)
        assert out.exists()

    def test_columns_normalised(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: _mock_toi_df())
        out = tmp_path / "toi.csv"
        mod.fetch_toi_table(out)
        rows = _read_csv(out)
        assert rows
        assert "toi" in rows[0]
        assert "tfopwg_disposition" in rows[0]

    def test_snr_column_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import Skills.fetch_tess_toi as mod
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: _mock_toi_df())
        out = tmp_path / "toi.csv"
        mod.fetch_toi_table(out)
        rows = _read_csv(out)
        assert rows
        assert "snr" in rows[0]
