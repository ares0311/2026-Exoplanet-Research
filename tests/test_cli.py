"""Tests for src/exo_toolkit/cli.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from exo_toolkit.cli import app, run_pipeline
from exo_toolkit.fetch import FetchProvenance
from exo_toolkit.schemas import (
    CandidateFeatures,
    CandidateScores,
    CandidateSignal,
    HypothesisPosterior,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_lc(n: int = 500) -> MagicMock:
    """Minimal LightCurve mock compatible with clean/search/vet."""
    time = np.linspace(2458000.0, 2458027.0, n)
    flux = np.ones(n)
    flux_err = np.full(n, 1e-4)
    lc = MagicMock()
    lc.time.jd = time
    lc.flux.value = flux
    lc.flux_err.value = flux_err
    return lc


def _make_provenance(
    *,
    cadence_seconds: float = 120.0,
    sectors: tuple[int, ...] = (1, 2, 3),
    pipeline: str = "SPOC",
) -> FetchProvenance:
    return FetchProvenance(
        target_id="TIC 0",
        mission="TESS",
        sectors_or_quarters=sectors,
        cadence_seconds=cadence_seconds,
        pipeline=pipeline,
        flux_column="pdcsap_flux",
        n_cadences=1000,
        time_baseline_days=81.0,
        fetched_at="2026-01-01T00:00:00+00:00",
    )


def _make_fetch_result(lc: Any, **prov_kwargs: Any) -> MagicMock:
    fr = MagicMock()
    fr.light_curve = lc
    fr.provenance = _make_provenance(**prov_kwargs)
    return fr


def _make_signal(period: float = 3.0) -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC0-001",
        mission="TESS",
        target_id="TIC 0",
        period_days=period,
        epoch_bjd=2458001.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        transit_count=9,
        snr=12.5,
    )


def _uniform_posterior() -> HypothesisPosterior:
    v = round(1.0 / 6, 10)
    remainder = round(1.0 - v * 5, 10)
    return HypothesisPosterior(
        planet_candidate=v,
        eclipsing_binary=v,
        background_eclipsing_binary=v,
        stellar_variability=v,
        instrumental_artifact=v,
        known_object=remainder,
    )


def _make_scores() -> CandidateScores:
    return CandidateScores(
        false_positive_probability=0.3,
        detection_confidence=0.7,
        novelty_score=0.5,
    )


# ---------------------------------------------------------------------------
# run_pipeline unit tests (mock fetch + full downstream pipeline)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def _patched_fetch(self, lc: Any) -> Any:
        def _fn(target_id: str, mission: str) -> Any:
            return _make_fetch_result(lc)

        return _fn

    def _patched_clean(self, lc: Any) -> Any:
        def _fn(light_curve: Any) -> Any:
            return MagicMock(light_curve=lc)

        return _fn

    def test_returns_list(self) -> None:
        lc = _mock_lc()
        with patch("exo_toolkit.cli.search_lightcurve", return_value=[]):
            result = run_pipeline(
                "TIC 0", "TESS",
                fetch_fn=self._patched_fetch(lc),
                clean_fn=self._patched_clean(lc),
            )
        assert isinstance(result, list)

    def test_empty_when_no_signals(self) -> None:
        lc = _mock_lc()
        with patch("exo_toolkit.cli.search_lightcurve", return_value=[]):
            result = run_pipeline(
                "TIC 0", "TESS",
                fetch_fn=self._patched_fetch(lc),
                clean_fn=self._patched_clean(lc),
            )
        assert result == []

    def test_one_row_per_signal(self) -> None:
        lc = _mock_lc()
        signals = [_make_signal(3.0), _make_signal(7.0)]
        posterior = _uniform_posterior()
        scores = _make_scores()

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=signals),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch(
                "exo_toolkit.cli.score_candidate",
                return_value=(posterior, scores),
            ),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
        ):
            result = run_pipeline(
                "TIC 0", "TESS",
                fetch_fn=self._patched_fetch(lc),
                clean_fn=self._patched_clean(lc),
            )

        assert len(result) == 2

    def test_row_contains_required_keys(self) -> None:
        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch(
                "exo_toolkit.cli.score_candidate",
                return_value=(posterior, scores),
            ),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="github_only_reproducibility",
            ),
        ):
            result = run_pipeline(
                "TIC 0", "TESS",
                fetch_fn=self._patched_fetch(lc),
                clean_fn=self._patched_clean(lc),
            )

        row = result[0]
        required = (
            "candidate_id", "period_days", "depth_ppm", "snr",
            "posterior", "scores", "pathway",
        )
        for key in required:
            assert key in row, f"Missing key: {key}"

    def test_posterior_sums_to_one(self) -> None:
        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch(
                "exo_toolkit.cli.score_candidate",
                return_value=(posterior, scores),
            ),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="github_only_reproducibility",
            ),
        ):
            result = run_pipeline(
                "TIC 0", "TESS",
                fetch_fn=self._patched_fetch(lc),
                clean_fn=self._patched_clean(lc),
            )

        p = result[0]["posterior"]
        total = sum(p.values())
        assert abs(total - 1.0) < 0.02

    def test_pipeline_error_propagates(self) -> None:
        def _bad_fetch(target_id: str, mission: str) -> Any:
            raise RuntimeError("network failure")

        with pytest.raises(RuntimeError, match="network failure"):
            run_pipeline("TIC 0", "TESS", fetch_fn=_bad_fetch)


# ---------------------------------------------------------------------------
# CLI command tests via CliRunner
# ---------------------------------------------------------------------------


class TestScanCommand:
    def _patch_pipeline(
        self, rows: list[dict[str, Any]]
    ) -> Any:
        return patch("exo_toolkit.cli.run_pipeline", return_value=rows)

    def test_no_candidates_exits_zero(self) -> None:
        with self._patch_pipeline([]):
            result = runner.invoke(app, ["TIC 0"])
        assert result.exit_code == 0, (
            f"exit={result.exit_code} output={result.output!r} exc={result.exception!r}"
        )
        assert "No transit candidates" in result.output

    def test_one_candidate_shown(self) -> None:
        rows = [
            {
                "candidate_id": "TIC0-001",
                "target_id": "TIC 0",
                "mission": "TESS",
                "period_days": 3.0,
                "epoch_bjd": 2458001.0,
                "duration_hours": 2.0,
                "depth_ppm": 1000.0,
                "transit_count": 9,
                "snr": 12.5,
                "provenance_score": 1.0,
                "posterior": {
                    "planet_candidate": 0.167,
                    "eclipsing_binary": 0.167,
                    "background_eclipsing_binary": 0.167,
                    "stellar_variability": 0.167,
                    "instrumental_artifact": 0.167,
                    "known_object": 0.165,
                },
                "scores": {
                    "false_positive_probability": 0.3,
                    "detection_confidence": 0.7,
                    "novelty_score": 0.5,
                },
                "pathway": "planet_hunters_discussion",
            }
        ]
        with self._patch_pipeline(rows):
            result = runner.invoke(app, ["TIC 0"])
        assert result.exit_code == 0
        assert "1 candidate signal" in result.output
        assert "planet_hunters_discussion" in result.output

    def test_invalid_mission_exits_nonzero(self) -> None:
        with self._patch_pipeline([]):
            result = runner.invoke(app, ["TIC 0", "--mission", "HST"])
        assert result.exit_code != 0

    def test_pipeline_exception_exits_2(self) -> None:
        with patch("exo_toolkit.cli.run_pipeline", side_effect=RuntimeError("fail")):
            result = runner.invoke(app, ["TIC 0"])
        assert result.exit_code == 2

    def test_output_json_written(self, tmp_path: Path) -> None:
        rows = [
            {
                "candidate_id": "TIC0-001",
                "target_id": "TIC 0",
                "mission": "TESS",
                "period_days": 3.0,
                "epoch_bjd": 2458001.0,
                "duration_hours": 2.0,
                "depth_ppm": 1000.0,
                "transit_count": 9,
                "snr": 12.5,
                "provenance_score": 1.0,
                "posterior": {
                    "planet_candidate": 0.167,
                    "eclipsing_binary": 0.167,
                    "background_eclipsing_binary": 0.167,
                    "stellar_variability": 0.167,
                    "instrumental_artifact": 0.167,
                    "known_object": 0.165,
                },
                "scores": {
                    "false_positive_probability": 0.3,
                    "detection_confidence": 0.7,
                    "novelty_score": 0.5,
                },
                "pathway": "github_only_reproducibility",
            }
        ]
        out_file = tmp_path / "results.json"
        with self._patch_pipeline(rows):
            result = runner.invoke(app, ["TIC 0", "--output", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 1
        assert data[0]["candidate_id"] == "TIC0-001"

    def test_mission_option_passed(self) -> None:
        captured: dict[str, Any] = {}

        def _spy(target_id: str, mission: str, **kwargs: Any) -> list[Any]:
            captured["mission"] = mission
            return []

        with patch("exo_toolkit.cli.run_pipeline", side_effect=_spy):
            runner.invoke(app, ["TIC 0", "--mission", "Kepler"])

        assert captured.get("mission") == "Kepler"

    def test_min_snr_passed(self) -> None:
        captured: dict[str, Any] = {}

        def _spy(target_id: str, mission: str, **kwargs: Any) -> list[Any]:
            captured["min_snr"] = kwargs.get("min_snr")
            return []

        with patch("exo_toolkit.cli.run_pipeline", side_effect=_spy):
            runner.invoke(app, ["TIC 0", "--min-snr", "8.5"])

        assert captured.get("min_snr") == pytest.approx(8.5)

    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TIC" in result.output


# ---------------------------------------------------------------------------
# Scorer / model-path option tests
# ---------------------------------------------------------------------------


class TestScorerOption:
    def test_default_scorer_is_bayesian(self) -> None:
        captured: dict[str, Any] = {}

        def _spy(target_id: str, mission: str, **kwargs: Any) -> list[Any]:
            captured["scorer"] = kwargs.get("scorer")
            return []

        with patch("exo_toolkit.cli.run_pipeline", side_effect=_spy):
            runner.invoke(app, ["TIC 0"])

        assert captured.get("scorer") == "bayesian"

    def test_scorer_option_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        def _spy(target_id: str, mission: str, **kwargs: Any) -> list[Any]:
            captured["scorer"] = kwargs.get("scorer")
            captured["model_path"] = kwargs.get("model_path")
            return []

        with patch("exo_toolkit.cli.run_pipeline", side_effect=_spy):
            runner.invoke(app, ["TIC 0", "--scorer", "xgboost", "--model-path", "/tmp/m.json"])

        assert captured.get("scorer") == "xgboost"
        assert captured.get("model_path") == Path("/tmp/m.json")

    def test_xgboost_without_model_path_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["TIC 0", "--scorer", "xgboost"])
        assert result.exit_code != 0

    def test_ensemble_without_model_path_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["TIC 0", "--scorer", "ensemble"])
        assert result.exit_code != 0

    def test_invalid_scorer_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["TIC 0", "--scorer", "neural_net"])
        assert result.exit_code != 0

    def test_run_pipeline_xgboost_adds_xgb_key(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from exo_toolkit.cli import run_pipeline

        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()

        mock_scorer = MagicMock()
        mock_scorer.predict_proba.return_value = 0.77

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
            patch("exo_toolkit.ml.xgboost_scorer.XGBoostScorer.load", return_value=mock_scorer),
        ):
            result = run_pipeline(
                "TIC 0",
                "TESS",
                scorer="xgboost",
                model_path=tmp_path / "model.json",
                fetch_fn=lambda *_: _make_fetch_result(lc),
                clean_fn=lambda *_: MagicMock(light_curve=lc),
            )

        assert len(result) == 1
        assert "xgb_planet_probability" in result[0]
        assert abs(result[0]["xgb_planet_probability"] - 0.77) < 1e-9


# ---------------------------------------------------------------------------
# Task 5: provenance_score flows through run_pipeline
# ---------------------------------------------------------------------------


class TestProvenanceScoreFlow:
    """Verify that compute_provenance_score is called and its output reaches the row."""

    def _run(self, lc: Any, **prov_kwargs: Any) -> list[dict[str, Any]]:
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()
        fetch_result = _make_fetch_result(lc, **prov_kwargs)

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
        ):
            return run_pipeline(
                "TIC 0",
                "TESS",
                fetch_fn=lambda *_: fetch_result,
                clean_fn=lambda *_: MagicMock(light_curve=lc),
            )

    def test_provenance_score_present_in_row(self) -> None:
        lc = _mock_lc()
        result = self._run(lc)
        assert "provenance_score" in result[0]

    def test_2min_spoc_3sectors_score_is_one(self) -> None:
        lc = _mock_lc()
        result = self._run(lc, cadence_seconds=120.0, sectors=(1, 2, 3), pipeline="SPOC")
        assert abs(result[0]["provenance_score"] - 1.0) < 1e-9

    def test_30min_single_sector_score_below_threshold(self) -> None:
        lc = _mock_lc()
        result = self._run(lc, cadence_seconds=1800.0, sectors=(1,), pipeline="QLP")
        assert result[0]["provenance_score"] < 0.80

    def test_provenance_score_passed_to_classify_pathway(self) -> None:
        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()
        fetch_result = _make_fetch_result(lc, cadence_seconds=120.0, sectors=(1, 2, 3))
        captured: list[dict[str, Any]] = []

        def _fake_classify(*args: Any, **kwargs: Any) -> str:
            captured.append(kwargs)
            return "planet_hunters_discussion"

        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch("exo_toolkit.cli.classify_submission_pathway", side_effect=_fake_classify),
        ):
            run_pipeline(
                "TIC 0",
                "TESS",
                fetch_fn=lambda *_: fetch_result,
                clean_fn=lambda *_: MagicMock(light_curve=lc),
            )

        assert "provenance_score" in captured[0]
        assert abs(captured[0]["provenance_score"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# CLI --version flag (Milestone 12m)
# ---------------------------------------------------------------------------


class TestCLIVersion:
    def test_version_flag_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_version_output_contains_semver(self) -> None:
        import re
        result = runner.invoke(app, ["--version"])
        assert re.search(r"\d+\.\d+\.\d+", result.output)

    def test_short_version_flag_works(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        import re
        assert re.search(r"\d+\.\d+\.\d+", result.output)

    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_scan_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["scan", "--help"])
        assert result.exit_code == 0

    def test_version_consistent_with_package(self) -> None:
        from exo_toolkit import __version__
        result = runner.invoke(app, ["--version"])
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# Richer JSON output metadata (Milestone 12n)
# ---------------------------------------------------------------------------


class TestOutputMetadata:
    def _run_pipeline_mocked(self, scorer: str = "bayesian") -> list[dict]:
        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()
        fetch_result = _make_fetch_result(lc, cadence_seconds=120.0, sectors=(1, 2, 3))
        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
        ):
            return run_pipeline(
                "TIC 0",
                "TESS",
                scorer=scorer,
                fetch_fn=lambda *_: fetch_result,
                clean_fn=lambda *_: MagicMock(light_curve=lc),
            )

    def test_row_contains_meta_key(self) -> None:
        rows = self._run_pipeline_mocked()
        assert len(rows) == 1
        assert "meta" in rows[0]

    def test_meta_toolkit_version_is_string(self) -> None:
        rows = self._run_pipeline_mocked()
        assert isinstance(rows[0]["meta"]["toolkit_version"], str)
        assert len(rows[0]["meta"]["toolkit_version"]) > 0

    def test_meta_run_at_is_iso8601(self) -> None:
        from datetime import datetime
        rows = self._run_pipeline_mocked()
        run_at = rows[0]["meta"]["run_at"]
        datetime.fromisoformat(run_at)  # raises if invalid

    def test_meta_scorer_matches_argument(self) -> None:
        rows = self._run_pipeline_mocked(scorer="bayesian")
        assert rows[0]["meta"]["scorer"] == "bayesian"

    def test_meta_git_commit_none_or_string(self) -> None:
        rows = self._run_pipeline_mocked()
        commit = rows[0]["meta"]["git_commit"]
        assert commit is None or isinstance(commit, str)

    def test_meta_features_available_is_list(self) -> None:
        rows = self._run_pipeline_mocked()
        assert isinstance(rows[0]["meta"]["features_available"], list)

    def test_meta_features_available_names_valid(self) -> None:
        rows = self._run_pipeline_mocked()
        valid_names = set(CandidateFeatures.model_fields.keys())
        for name in rows[0]["meta"]["features_available"]:
            assert name in valid_names

    def test_json_output_contains_meta(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        rows = self._run_pipeline_mocked()
        assert "meta" in rows[0]
        out.write_text(json.dumps(rows))
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert "meta" in data[0]

    def test_meta_present_with_bayesian_scorer(self) -> None:
        rows = self._run_pipeline_mocked(scorer="bayesian")
        assert "meta" in rows[0]

    def test_meta_scorer_field_correct_for_bayesian(self) -> None:
        rows = self._run_pipeline_mocked(scorer="bayesian")
        assert rows[0]["meta"]["scorer"] == "bayesian"


# ---------------------------------------------------------------------------
# Calibration integration (Milestone 13i)
# ---------------------------------------------------------------------------


class TestCalibrationIntegration:
    """Tests for calibration_path parameter in run_pipeline."""

    def _run_with_mocks(self, **kwargs: Any) -> list[dict[str, Any]]:
        lc = _mock_lc()
        signal = _make_signal()
        posterior = _uniform_posterior()
        scores = _make_scores()
        fetch_result = _make_fetch_result(lc)
        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[signal]),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
        ):
            return run_pipeline(
                "TIC 0",
                "TESS",
                fetch_fn=lambda *_: fetch_result,
                clean_fn=lambda *_: MagicMock(light_curve=lc),
                **kwargs,
            )

    def test_no_calibration_path_no_calibrated_posterior_key(self) -> None:
        rows = self._run_with_mocks()
        assert "calibrated_posterior" not in rows[0]

    def test_calibrated_posterior_present_when_path_given(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch("exo_toolkit.cli.apply_calibration", return_value=mock_cal_post),
        ):
            rows = self._run_with_mocks(calibration_path=tmp_path / "cal.json")
        assert "calibrated_posterior" in rows[0]

    def test_calibrated_posterior_is_dict(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch("exo_toolkit.cli.apply_calibration", return_value=mock_cal_post),
        ):
            rows = self._run_with_mocks(calibration_path=tmp_path / "cal.json")
        assert isinstance(rows[0]["calibrated_posterior"], dict)

    def test_calibrated_posterior_has_six_hypothesis_keys(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch("exo_toolkit.cli.apply_calibration", return_value=mock_cal_post),
        ):
            rows = self._run_with_mocks(calibration_path=tmp_path / "cal.json")
        expected = {
            "planet_candidate", "eclipsing_binary", "background_eclipsing_binary",
            "stellar_variability", "instrumental_artifact", "known_object",
        }
        assert set(rows[0]["calibrated_posterior"].keys()) == expected

    def test_calibrated_posterior_sums_to_one(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch("exo_toolkit.cli.apply_calibration", return_value=mock_cal_post),
        ):
            rows = self._run_with_mocks(calibration_path=tmp_path / "cal.json")
        total = sum(rows[0]["calibrated_posterior"].values())
        assert abs(total - 1.0) < 0.02

    def test_load_calibration_called_once_with_path(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        cal_path = tmp_path / "cal.json"
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal) as mock_load,
            patch("exo_toolkit.cli.apply_calibration", return_value=mock_cal_post),
        ):
            self._run_with_mocks(calibration_path=cal_path)
        mock_load.assert_called_once_with(cal_path)

    def test_apply_calibration_called_per_signal(self, tmp_path: Path) -> None:
        lc = _mock_lc()
        signals = [_make_signal(3.0), _make_signal(7.0)]
        posterior = _uniform_posterior()
        scores = _make_scores()
        fetch_result = _make_fetch_result(lc)
        mock_cal = MagicMock()
        mock_cal_post = _uniform_posterior()
        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=signals),
            patch(
                "exo_toolkit.cli.vet_signal",
                return_value=MagicMock(features=CandidateFeatures()),
            ),
            patch("exo_toolkit.cli.score_candidate", return_value=(posterior, scores)),
            patch(
                "exo_toolkit.cli.classify_submission_pathway",
                return_value="planet_hunters_discussion",
            ),
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch(
                "exo_toolkit.cli.apply_calibration", return_value=mock_cal_post
            ) as mock_apply,
        ):
            run_pipeline(
                "TIC 0",
                "TESS",
                fetch_fn=lambda *_: fetch_result,
                clean_fn=lambda *_: MagicMock(light_curve=lc),
                calibration_path=tmp_path / "cal.json",
            )
        assert mock_apply.call_count == 2

    def test_calibration_load_error_propagates(self, tmp_path: Path) -> None:
        with (
            patch("exo_toolkit.cli.load_calibration", side_effect=ValueError("bad file")),
            pytest.raises(ValueError, match="bad file"),
        ):
            self._run_with_mocks(calibration_path=tmp_path / "cal.json")

    def test_raw_posterior_and_calibrated_can_differ(self, tmp_path: Path) -> None:
        mock_cal = MagicMock()
        shifted_post = HypothesisPosterior(
            planet_candidate=0.50,
            eclipsing_binary=0.10,
            background_eclipsing_binary=0.10,
            stellar_variability=0.10,
            instrumental_artifact=0.10,
            known_object=0.10,
        )
        with (
            patch("exo_toolkit.cli.load_calibration", return_value=mock_cal),
            patch("exo_toolkit.cli.apply_calibration", return_value=shifted_post),
        ):
            rows = self._run_with_mocks(calibration_path=tmp_path / "cal.json")
        assert rows[0]["calibrated_posterior"]["planet_candidate"] == pytest.approx(0.50)

    def test_no_apply_calibration_when_no_signals(self) -> None:
        lc = _mock_lc()
        with (
            patch("exo_toolkit.cli.search_lightcurve", return_value=[]),
            patch("exo_toolkit.cli.load_calibration", return_value=MagicMock()),
            patch("exo_toolkit.cli.apply_calibration") as mock_apply,
        ):
            result = run_pipeline(
                "TIC 0",
                "TESS",
                fetch_fn=lambda *_: _make_fetch_result(lc),
                clean_fn=lambda *_: MagicMock(light_curve=lc),
                calibration_path=Path("/tmp/cal.json"),
            )
        assert result == []
        mock_apply.assert_not_called()
