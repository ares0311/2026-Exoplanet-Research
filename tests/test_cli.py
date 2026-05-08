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


def _make_fetch_result(lc: Any) -> MagicMock:
    fr = MagicMock()
    fr.light_curve = lc
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
                fetch_fn=lambda *_: MagicMock(light_curve=lc),
                clean_fn=lambda *_: MagicMock(light_curve=lc),
            )

        assert len(result) == 1
        assert "xgb_planet_probability" in result[0]
        assert abs(result[0]["xgb_planet_probability"] - 0.77) < 1e-9
