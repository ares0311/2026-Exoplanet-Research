"""Tests for Skills/feature_importance_ranker.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from feature_importance_ranker import (
    FeatureImportanceEntry,
    format_feature_importance,
    load_and_rank,
    rank_features,
)

_SAMPLE_METADATA = {
    "feature_names": ["snr_score", "depth_score", "period_ratio"],
    "training_result": {
        "feature_importance": {
            "snr_score": 0.5,
            "depth_score": 0.3,
            "period_ratio": 0.2,
        }
    },
}


def test_valid_metadata_ok():
    result = rank_features(_SAMPLE_METADATA)
    assert result.flag == "OK"


def test_top_rank_has_highest_importance():
    result = rank_features(_SAMPLE_METADATA)
    top = result.entries[0]
    assert top.name == "snr_score"
    assert top.importance == 0.5


def test_ranks_are_one_based_contiguous():
    result = rank_features(_SAMPLE_METADATA)
    ranks = [e.rank for e in result.entries]
    assert ranks == list(range(1, len(result.entries) + 1))


def test_n_features_matches_dict_length():
    result = rank_features(_SAMPLE_METADATA)
    assert result.n_features == 3


def test_empty_feature_importance_returns_empty():
    meta = {"training_result": {"feature_importance": {}}}
    result = rank_features(meta)
    assert result.flag == "EMPTY"
    assert result.n_features == 0


def test_missing_training_result_key_returns_invalid():
    result = rank_features({"feature_names": ["a", "b"]})
    assert result.flag == "INVALID"


def test_format_has_feature_name():
    result = rank_features(_SAMPLE_METADATA)
    md = format_feature_importance(result)
    assert "snr_score" in md


def test_format_has_rank_column():
    result = rank_features(_SAMPLE_METADATA)
    md = format_feature_importance(result)
    assert "Rank" in md


def test_importance_type_stored_in_result():
    result = rank_features(_SAMPLE_METADATA, importance_type="cover")
    assert result.importance_type == "cover"


def test_all_importances_non_negative():
    result = rank_features(_SAMPLE_METADATA)
    for entry in result.entries:
        assert entry.importance >= 0.0


def test_feature_importance_result_frozen():
    result = rank_features(_SAMPLE_METADATA)
    try:
        result.flag = "MODIFIED"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_feature_importance_entry_frozen():
    entry = FeatureImportanceEntry(name="x", importance=0.1, rank=1)
    try:
        entry.rank = 99  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_load_and_rank_from_temp_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as fh:
        json.dump(_SAMPLE_METADATA, fh)
        tmp_path = Path(fh.name)
    try:
        result = load_and_rank(tmp_path)
        assert result.flag == "OK"
        assert result.n_features == 3
    finally:
        tmp_path.unlink(missing_ok=True)
