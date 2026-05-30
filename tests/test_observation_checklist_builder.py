"""Tests for Skills/observation_checklist_builder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_checklist_builder import (  # noqa: E402
    ChecklistItem,
    ObsChecklist,
    build_observation_checklist,
    format_observation_checklist,
)


def test_returns_dataclass():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert isinstance(r, ObsChecklist)


def test_flag_ok():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert r.flag == "OK"


def test_target_stored():
    r = build_observation_checklist("TIC-999", 3.0, 2458000.0, 2.0)
    assert r.target == "TIC-999"


def test_pre_items_count():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert r.n_pre_items == 6


def test_during_items_count():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert r.n_during_items == 4


def test_post_items_count():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert r.n_post_items == 4


def test_total_items():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    assert len(r.items) == r.n_pre_items + r.n_during_items + r.n_post_items


def test_items_are_checklist_items():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    for item in r.items:
        assert isinstance(item, ChecklistItem)
        assert item.phase in ("pre", "during", "post")


def test_epoch_in_pre_text():
    r = build_observation_checklist("TIC-1", 3.14, 2458123.4567, 2.0)
    pre_texts = " ".join(item.text for item in r.items if item.phase == "pre")
    assert "2458123.4567" in pre_texts


def test_format_returns_string():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    s = format_observation_checklist(r)
    assert isinstance(s, str)


def test_format_contains_target():
    r = build_observation_checklist("MyTarget", 3.0, 2458000.0, 2.0)
    s = format_observation_checklist(r)
    assert "MyTarget" in s


def test_format_contains_phase_sections():
    r = build_observation_checklist("TIC-1", 3.0, 2458000.0, 2.0)
    s = format_observation_checklist(r)
    assert "Pre-Observation" in s
    assert "During" in s
    assert "Post-Observation" in s
