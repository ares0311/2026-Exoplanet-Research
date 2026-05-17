"""Tests for Skills.bootstrap_uncertainty."""
from __future__ import annotations

import pytest
from Skills.bootstrap_uncertainty import (
    BootstrapResult,
    bootstrap_uncertainty,
    format_bootstrap_result,
)


def _row(**kw) -> dict:
    base = {"tic_id": 1, "best_fpp": 0.2, "rank_score": 0.7}
    base.update(kw)
    return base


class TestBootstrapUncertainty:
    def test_returns_bootstrap_result(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=50)
        assert isinstance(r, BootstrapResult)

    def test_n_samples_stored(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=100)
        assert r.n_samples == 100

    def test_fpp_mean_in_range(self) -> None:
        r = bootstrap_uncertainty(_row(best_fpp=0.3), n_samples=100)
        assert 0.0 <= r.fpp_mean <= 1.0

    def test_ci_interval_ordered(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=200)
        assert r.fpp_ci_low <= r.fpp_ci_high

    def test_ci_level_stored(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=50, ci_level=0.90)
        assert r.ci_level == pytest.approx(0.90)

    def test_rank_score_mean_in_range(self) -> None:
        r = bootstrap_uncertainty(_row(rank_score=0.8), n_samples=100)
        assert 0.0 <= r.rank_score_mean <= 1.0

    def test_reproducible_with_seed(self) -> None:
        r1 = bootstrap_uncertainty(_row(), n_samples=100, seed=7)
        r2 = bootstrap_uncertainty(_row(), n_samples=100, seed=7)
        assert r1.fpp_mean == pytest.approx(r2.fpp_mean)

    def test_custom_score_fn(self) -> None:
        def score_fn(row):
            return row.get("best_fpp", 0.5), row.get("rank_score", 0.5)
        r = bootstrap_uncertainty(_row(), n_samples=50, score_fn=score_fn)
        assert isinstance(r, BootstrapResult)

    def test_fpp_std_nonnegative(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=100)
        assert r.fpp_std >= 0.0

    def test_rank_score_std_nonnegative(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=100)
        assert r.rank_score_std >= 0.0

    def test_high_fpp_row(self) -> None:
        r = bootstrap_uncertainty(_row(best_fpp=0.99), n_samples=50)
        assert r.fpp_mean > 0.9


class TestFormatBootstrapResult:
    def test_returns_string(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=50)
        assert isinstance(format_bootstrap_result(r), str)

    def test_contains_samples(self) -> None:
        r = bootstrap_uncertainty(_row(), n_samples=50)
        assert "50" in format_bootstrap_result(r)
