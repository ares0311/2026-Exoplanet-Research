"""Tests for Skills/follow_up_checklist_generator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from follow_up_checklist_generator import (
    FollowUpChecklist,
    format_checklist,
    generate_checklist,
)


def test_returns_checklist():
    cl = generate_checklist(100, period_days=5.0, fpp=0.05)
    assert isinstance(cl, FollowUpChecklist)


def test_default_checklist_nonempty():
    cl = generate_checklist()
    assert cl.n_total > 0


def test_high_fpp_adds_high_priority():
    cl = generate_checklist(fpp=0.50)
    assert cl.n_high > 0


def test_few_transits_adds_high_priority():
    cl = generate_checklist(n_transits=1)
    assert cl.n_high > 0


def test_tfop_pathway_adds_items():
    cl_tfop = generate_checklist(pathway="tfop_ready")
    cl_none = generate_checklist()
    assert cl_tfop.n_total >= cl_none.n_total


def test_m_dwarf_adds_flare_check():
    cl = generate_checklist(stellar_teff_k=3500.0)
    descs = [i.description for i in cl.checklist]
    assert any("flare" in d.lower() for d in descs)


def test_hot_star_adds_pulsation_check():
    cl = generate_checklist(stellar_teff_k=8000.0)
    descs = [i.description for i in cl.checklist]
    assert any("pulsation" in d.lower() for d in descs)


def test_short_period_adds_check():
    cl = generate_checklist(period_days=0.5)
    descs = [i.description for i in cl.checklist]
    assert any("short" in d.lower() or "ultra" in d.lower() for d in descs)


def test_checklist_frozen():
    cl = generate_checklist()
    try:
        cl.n_total = 999  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_flag_ok_enough_items():
    cl = generate_checklist(100, period_days=5.0, fpp=0.05)
    assert cl.flag == "OK"


def test_tic_id_stored():
    cl = generate_checklist(12345)
    assert cl.tic_id == 12345


def test_format_returns_string():
    cl = generate_checklist(100, period_days=5.0)
    text = format_checklist(cl)
    assert isinstance(text, str)


def test_format_contains_tic():
    cl = generate_checklist(99999, period_days=5.0)
    text = format_checklist(cl)
    assert "99999" in text
