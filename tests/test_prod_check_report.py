"""Print the live prod-check report. Run with -s to view.

Kept separate from the assertions in ``test_prod_check.py`` so the gate's
human-readable output can be inspected without a console-script reinstall.
"""
from __future__ import annotations

from exo_toolkit.prod_check import main


def test_emit_report() -> None:
    assert main([]) in (0, 1)
