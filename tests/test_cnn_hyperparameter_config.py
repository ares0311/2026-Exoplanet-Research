"""Tests for Skills/cnn_hyperparameter_config.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from cnn_hyperparameter_config import (
    HyperparamGrid,
    default_grid,
    format_grid_summary,
    generate_candidates,
    load_grid,
    save_grid,
)


def test_default_grid_returns_grid():
    g = default_grid()
    assert isinstance(g, HyperparamGrid)


def test_default_has_candidates():
    g = default_grid()
    assert len(g.learning_rates) > 0
    assert len(g.dropout_rates) > 0


def test_generate_count():
    g = default_grid()
    cands = generate_candidates(g)
    expected = (
        len(g.learning_rates) * len(g.dropout_rates) *
        len(g.conv_filters) * len(g.fc_units) * len(g.batch_sizes)
    )
    assert len(cands) == expected


def test_candidate_ids_sequential():
    g = default_grid()
    cands = generate_candidates(g)
    assert [c.candidate_id for c in cands] == list(range(len(cands)))


def test_candidate_is_frozen():
    g = default_grid()
    c = generate_candidates(g)[0]
    try:
        c.learning_rate = 99.0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_save_load_roundtrip(tmp_path):
    g = default_grid()
    p = tmp_path / "grid.json"
    save_grid(g, p)
    g2 = load_grid(p)
    assert g2.learning_rates == g.learning_rates
    assert g2.n_epochs == g.n_epochs


def test_save_creates_parent(tmp_path):
    g = default_grid()
    p = tmp_path / "sub" / "grid.json"
    save_grid(g, p)
    assert p.exists()


def test_format_summary_string():
    g = default_grid()
    s = format_grid_summary(g)
    assert isinstance(s, str)
    assert "Hyperparameter" in s


def test_format_shows_total():
    g = default_grid()
    s = format_grid_summary(g)
    cands = generate_candidates(g)
    assert str(len(cands)) in s


def test_grid_frozen():
    g = default_grid()
    try:
        g.n_epochs = 0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_minimal_grid():
    g = HyperparamGrid(
        learning_rates=(1e-3,),
        dropout_rates=(0.5,),
        conv_filters=(32,),
        fc_units=(64,),
        batch_sizes=(32,),
        n_epochs=5,
    )
    cands = generate_candidates(g)
    assert len(cands) == 1


def test_candidates_cover_all_lrs():
    g = default_grid()
    cands = generate_candidates(g)
    observed_lrs = {c.learning_rate for c in cands}
    assert observed_lrs == set(g.learning_rates)
