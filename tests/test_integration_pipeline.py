"""End-to-end pipeline integration tests using mocked I/O (no network).

These tests exercise the full run_pipeline() wiring — fetch → clean → search →
vet → score → classify — with the I/O-heavy steps mocked so no network or
real light curve is needed.  The scoring and pathway logic runs for real.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from exo_toolkit.cli import run_pipeline
from exo_toolkit.fetch import FetchProvenance
from exo_toolkit.schemas import CandidateFeatures, CandidateSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance(**kwargs: Any) -> FetchProvenance:
    defaults: dict[str, Any] = {
        "target_id": "TIC 0",
        "mission": "TESS",
        "sectors_or_quarters": (1, 2, 3),
        "cadence_seconds": 120.0,
        "pipeline": "SPOC",
        "flux_column": "pdcsap_flux",
        "n_cadences": 600,
        "time_baseline_days": 30.0,
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }
    defaults.update(kwargs)
    return FetchProvenance(**defaults)


def _make_signal(**kwargs: Any) -> CandidateSignal:
    defaults: dict[str, Any] = {
        "candidate_id": "TIC0-001",
        "mission": "TESS",
        "target_id": "TIC 0",
        "period_days": 5.0,
        "epoch_bjd": 2458001.0,
        "duration_hours": 2.0,
        "depth_ppm": 1000.0,
        "transit_count": 3,
        "snr": 12.5,
    }
    defaults.update(kwargs)
    return CandidateSignal(**defaults)


def _make_lc() -> MagicMock:
    lc = MagicMock()
    lc.time.jd = np.linspace(2458000.0, 2458030.0, 600)
    lc.flux.value = np.ones(600)
    lc.flux_err.value = np.full(600, 1e-4)
    return lc


def _make_fetch_result(lc: Any, **prov_kwargs: Any) -> MagicMock:
    fr = MagicMock()
    fr.light_curve = lc
    fr.provenance = _make_provenance(**prov_kwargs)
    return fr


def _run_with_mocks(
    signals: list[CandidateSignal] | None = None,
    features: CandidateFeatures | None = None,
    **pipeline_kwargs: Any,
) -> list[dict]:
    """Run run_pipeline() with search + vet mocked; scoring + pathway are real."""
    if signals is None:
        signals = [_make_signal()]
    if features is None:
        features = CandidateFeatures(
            log_snr_score=0.8,
            transit_count_score=0.7,
            odd_even_mismatch_score=0.05,
            secondary_eclipse_score=0.05,
            contamination_score=0.05,
        )
    lc = _make_lc()
    fr = _make_fetch_result(lc)
    with (
        patch("exo_toolkit.cli.search_lightcurve", return_value=signals),
        patch("exo_toolkit.cli.vet_signal", return_value=MagicMock(features=features)),
    ):
        return run_pipeline(
            "TIC 0",
            "TESS",
            fetch_fn=lambda *_: fr,
            clean_fn=lambda *_: MagicMock(light_curve=lc),
            **pipeline_kwargs,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    def test_returns_nonempty_list_for_transit(self) -> None:
        rows = _run_with_mocks()
        assert len(rows) >= 1

    def test_row_has_required_keys(self) -> None:
        rows = _run_with_mocks()
        row = rows[0]
        for key in ("candidate_id", "pathway", "scores", "posterior"):
            assert key in row, f"missing key: {key}"

    def test_posterior_sums_to_one(self) -> None:
        rows = _run_with_mocks()
        p = rows[0]["posterior"]
        total = sum(p.values())
        assert abs(total - 1.0) < 0.01

    def test_fpp_in_unit_interval(self) -> None:
        rows = _run_with_mocks()
        fpp = rows[0]["scores"]["false_positive_probability"]
        assert 0.0 <= fpp <= 1.0

    def test_pathway_is_valid_literal(self) -> None:
        valid = {
            "known_object_annotation", "tfop_ready", "planet_hunters_discussion",
            "kepler_archive_candidate", "github_only_reproducibility",
            "paper_or_preprint_candidate",
        }
        rows = _run_with_mocks()
        assert rows[0]["pathway"] in valid

    def test_scorer_bayesian_no_model_path(self) -> None:
        rows = _run_with_mocks(scorer="bayesian")
        assert len(rows) >= 1

    def test_unknown_scorer_raises(self) -> None:
        with pytest.raises(ValueError):
            _run_with_mocks(scorer="unknown_scorer")

    def test_xgboost_without_model_path_raises(self) -> None:
        with pytest.raises(ValueError):
            _run_with_mocks(scorer="xgboost", model_path=None)

    def test_provenance_score_in_unit_interval(self) -> None:
        rows = _run_with_mocks()
        assert "provenance_score" in rows[0]
        ps = rows[0]["provenance_score"]
        assert 0.0 <= ps <= 1.0

    def test_no_signals_returns_empty_list(self) -> None:
        rows = _run_with_mocks(signals=[])
        assert rows == []
