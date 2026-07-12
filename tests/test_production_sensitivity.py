from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from Skills.production_sensitivity import (
    build_trials,
    inject_flux,
    match_recovery,
    scoped_output_path,
    select_quarter_products,
    summarize_curves,
)


def _config() -> dict[str, object]:
    return {
        "backgrounds": [
            {"target_id": "KIC 1", "background_label": "negative"},
            {"target_id": "KIC 2", "background_label": "positive"},
        ],
        "injection_grid": {
            "period_days": [3.0, 30.0],
            "depth_ppm": [500.0, 2000.0],
            "duration_hours": [2.0],
            "epoch_phase": 0.25,
        },
    }


def test_build_trials_expands_grid_and_shards_without_overlap() -> None:
    all_trials = build_trials(_config())
    shard_zero = build_trials(_config(), shard_index=0, shard_count=2)
    shard_one = build_trials(_config(), shard_index=1, shard_count=2)
    assert len(all_trials) == 8
    assert {trial.index for trial in shard_zero}.isdisjoint(
        {trial.index for trial in shard_one}
    )
    assert {trial.index for trial in all_trials} == {
        trial.index for trial in shard_zero + shard_one
    }


def test_invalid_shard_fails_closed() -> None:
    with pytest.raises(ValueError):
        build_trials(_config(), shard_index=2, shard_count=2)


def test_inject_flux_applies_exact_depth_and_returns_copy() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    flux = np.ones_like(time)
    result = inject_flux(
        time,
        flux,
        period_days=2.0,
        epoch_bjd=0.0,
        duration_hours=4.8,
        depth_ppm=10_000.0,
    )
    assert result is not flux
    assert np.min(result) == pytest.approx(0.99)
    assert np.max(result) == pytest.approx(1.0)


def test_match_recovery_accepts_period_aliases() -> None:
    half = {"period_days": 5.0, "snr": 10.0}
    assert match_recovery([half], 10.0) is half
    assert match_recovery([{"period_days": 7.0}], 10.0) is None


def test_summarize_curves_preserves_sample_counts_and_model_context() -> None:
    results = [
        {
            "period_days": 3.0,
            "depth_ppm": 500.0,
            "duration_hours": 2.0,
            "background_label": "negative",
            "recovered": True,
            "cnn_planet_probability": 0.8,
            "full_ensemble_planet_probability": 0.7,
        },
        {
            "period_days": 3.0,
            "depth_ppm": 2000.0,
            "duration_hours": 2.0,
            "background_label": "negative",
            "recovered": False,
            "cnn_planet_probability": None,
            "full_ensemble_planet_probability": None,
        },
    ]
    curves = summarize_curves(results)
    assert curves["overall"] == {"trials": 2, "recovered": 1, "recovery_rate": 0.5}
    assert curves["by_period_days"][0]["mean_cnn_planet_probability"] == pytest.approx(0.8)


def test_scoped_output_path_is_collision_free() -> None:
    path = Path("reports/sensitivity.json")
    assert scoped_output_path(path, shard_index=0, shard_count=1) == path
    assert scoped_output_path(path, shard_index=2, shard_count=4) == Path(
        "reports/sensitivity.shard2of4.json"
    )


def test_quarter_filter_keeps_provenance_paths_aligned() -> None:
    class Curve:
        def __init__(self, quarter: int) -> None:
            self.meta = {"QUARTER": quarter}

    paths = [Path("q1.fits"), Path("q2.fits"), Path("q3.fits")]
    selected_paths, selected_curves = select_quarter_products(
        paths, [Curve(1), Curve(2), Curve(3)], {1}
    )
    assert selected_paths == [Path("q1.fits")]
    assert [curve.meta["QUARTER"] for curve in selected_curves] == [1]


def test_committed_config_has_two_frozen_eval_backgrounds() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    config = json.loads(
        (repo_root / "configs/production_sensitivity_v1.json").read_text(encoding="utf-8")
    )
    assert config["source_dataset_id"] == "t1_1_kepler_master_frozen_eval"
    assert len(config["backgrounds"]) == 2
    assert all(background["quarters"] == [1] for background in config["backgrounds"])
    assert all(background["max_baseline_days"] == 30.0 for background in config["backgrounds"])
    assert config["pipeline"]["period_max"] == 20.0
    assert "full_ensemble_v0.2.43" in config["model_ids"]
    assert len(build_trials(config)) == 36


def test_committed_evidence_is_complete_and_provenance_bounded() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    evidence = json.loads(
        (repo_root / "artifacts/manifests/production_sensitivity_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["toolkit_version"] == "0.2.43"
    assert evidence["suite_id"] == "production_sensitivity_v1"
    assert evidence["source_dataset_id"] == "t1_1_kepler_master_frozen_eval"
    assert "benchmark_cnn_v1" in evidence["model_ids"]
    assert evidence["curves"]["overall"] == {
        "trials": 36,
        "recovered": 23,
        "recovery_rate": 23 / 36,
    }
    assert evidence["failures"] == []
    assert all(
        background["sectors_or_quarters"] == [1]
        and len(background["raw_uris"]) == 1
        for background in evidence["backgrounds"]
    )
