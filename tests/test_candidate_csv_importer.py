"""Tests for candidate_csv_importer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_csv_importer import (
    format_import_result,
    import_candidates_csv,
)

_CSV_GOOD = (
    "tic_id,period_days,epoch_bjd,depth_ppm,duration_hours,snr\n"
    "150428135,37.42,2458000.0,5000.0,3.5,12.3\n"
    "260004324,20.1,2458100.5,,2.0,\n"
)
_CSV_BAD_PERIOD = "tic_id,period_days,epoch_bjd\n100,0.0,2458000.0\n"
_CSV_MISSING_COL = "tic_id,epoch_bjd\n100,2458000.0\n"


class TestImportCandidatesCSV:
    def test_result_frozen(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        try:
            r.n_imported = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_candidate_frozen(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        if r.candidates:
            c = r.candidates[0]
            try:
                c.tic_id = 99  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_missing_file_invalid(self, tmp_path):
        r = import_candidates_csv(tmp_path / "nonexistent.csv")
        assert r.flag == "INVALID"

    def test_good_csv_ok(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        assert r.flag == "OK"
        assert r.n_imported == 2

    def test_n_rows_read(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        assert r.n_rows_read == 2

    def test_tic_id_parsed(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        assert r.candidates[0].tic_id == 150428135

    def test_period_parsed(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        assert abs(r.candidates[0].period_days - 37.42) < 1e-5

    def test_optional_cols_none_when_empty(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        c = r.candidates[1]
        assert c.depth_ppm is None

    def test_bad_period_skipped(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_BAD_PERIOD)
        r = import_candidates_csv(p)
        assert r.n_skipped >= 1

    def test_missing_required_col_invalid(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_MISSING_COL)
        r = import_candidates_csv(p)
        assert r.flag == "INVALID"

    def test_empty_file_empty_flag(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("tic_id,period_days,epoch_bjd\n")
        r = import_candidates_csv(p)
        assert r.flag in ("EMPTY", "OK")

    def test_source_file_recorded(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        assert str(p) in r.candidates[0].source_file

    def test_format_returns_string(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text(_CSV_GOOD)
        r = import_candidates_csv(p)
        s = format_import_result(r)
        assert isinstance(s, str)
        assert "Import" in s
