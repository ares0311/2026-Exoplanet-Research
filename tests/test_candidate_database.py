"""Tests for Skills.candidate_database."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from Skills.candidate_database import CandidateDatabase


def _row(**kw: object) -> dict:
    base = {
        "tic_id": 1,
        "period_days": 5.0,
        "fpp": 0.05,
        "pathway": "tfop_ready",
        "scorer": "bayesian",
    }
    base.update(kw)
    return base


def _provenanced(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate_id": "TESS-TIC1-5d",
        "source_dataset_id": "canonical_tess_eval_v1",
        "target_id": "TIC 1",
        "mission": "TESS",
        "time_window": "sectors 1-2",
        "raw_uri": "mast:TESS/product.fits",
        "preprocess_version": "clean-v1",
        "candidate_generator": "BLS",
        "candidate_generator_params": {"min_snr": 5.0},
        "model_versions": {"bayesian": "0.2.31"},
        "model_scores": {"fpp": 0.05},
        "calibrated_scores": {},
        "score_quantiles": {},
        "injection_context": {"status": "pending"},
        "nearest_known_artifacts": [],
        "regeneration_command": "exo 'TIC 1' --mission TESS",
    }
    record.update(updates)
    return record


class TestCandidateDatabase:
    def test_insert_returns_id(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        row_id = db.insert(_row())
        assert isinstance(row_id, int) and row_id >= 1
        db.close()

    def test_latest_returns_none_for_unknown(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        assert db.latest(99999) is None
        db.close()

    def test_latest_returns_most_recent(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1, fpp=0.10))
        db.insert(_row(tic_id=1, fpp=0.05))
        row = db.latest(1)
        assert row is not None
        assert row["fpp"] == pytest.approx(0.05)
        db.close()

    def test_history_returns_all_rows(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1, fpp=0.10))
        db.insert(_row(tic_id=1, fpp=0.05))
        hist = db.history(1)
        assert len(hist) == 2
        db.close()

    def test_history_oldest_first(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1, fpp=0.20))
        db.insert(_row(tic_id=1, fpp=0.05))
        hist = db.history(1)
        assert hist[0]["fpp"] == pytest.approx(0.20)
        db.close()

    def test_all_latest_returns_one_per_tic(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1, fpp=0.05))
        db.insert(_row(tic_id=2, fpp=0.10))
        rows = db.all_latest()
        assert len(rows) == 2
        db.close()

    def test_delete_removes_rows(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1))
        db.insert(_row(tic_id=1))
        n = db.delete(1)
        assert n == 2
        assert db.latest(1) is None
        db.close()

    def test_count_correct(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=1))
        db.insert(_row(tic_id=2))
        assert db.count() == 2
        db.close()

    def test_export_csv_creates_file(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row())
        out = db.export_csv(tmp_path / "out.csv")
        assert out.exists()
        db.close()

    def test_export_csv_contains_tic_id(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert(_row(tic_id=99))
        db.export_csv(tmp_path / "out.csv")
        text = (tmp_path / "out.csv").read_text()
        assert "99" in text
        db.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        with CandidateDatabase(tmp_path / "c.sqlite3") as db:
            db.insert(_row())
            assert db.count() == 1

    def test_optional_fields_none(self, tmp_path: Path) -> None:
        db = CandidateDatabase(tmp_path / "c.sqlite3")
        db.insert({"tic_id": 1})
        row = db.latest(1)
        assert row is not None
        assert row["period_days"] is None
        db.close()

    def test_insert_provenanced_round_trips_contract(self, tmp_path: Path) -> None:
        with CandidateDatabase(tmp_path / "c.sqlite3") as db:
            row_id = db.insert_provenanced(_provenanced())
            history = db.candidate_history("TESS-TIC1-5d")

            assert row_id >= 1
            assert db.provenanced_count() == 1
            assert len(history) == 1
            assert history[0].candidate_id == "TESS-TIC1-5d"
            assert history[0].source_dataset_id == "canonical_tess_eval_v1"
            assert history[0].candidate_generator_params == {"min_snr": 5.0}

    def test_provenanced_history_is_append_only(self, tmp_path: Path) -> None:
        with CandidateDatabase(tmp_path / "c.sqlite3") as db:
            db.insert_provenanced(_provenanced(review_status="unreviewed"))
            db.insert_provenanced(
                _provenanced(
                    review_status="follow_up_worthy",
                    review_notes="independent review passed",
                )
            )

            history = db.candidate_history("TESS-TIC1-5d")
            assert [row.review_status for row in history] == [
                "unreviewed",
                "follow_up_worthy",
            ]

    def test_provenanced_insert_rejects_missing_dataset_id(
        self, tmp_path: Path
    ) -> None:
        record = _provenanced()
        record.pop("source_dataset_id")
        with (
            CandidateDatabase(tmp_path / "c.sqlite3") as db,
            pytest.raises(ValidationError),
        ):
            db.insert_provenanced(record)
