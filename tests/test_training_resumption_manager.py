"""Tests for Skills/training_resumption_manager.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from training_resumption_manager import (
    CheckpointInfo,
    ResumptionState,
    find_latest_checkpoint,
    format_resumption_state,
    plan_resumption,
)


def _write_checkpoint(d: Path, epoch: int, val_auc: float = 0.9) -> Path:
    p = d / f"epoch_{epoch:03d}.json"
    p.write_text(json.dumps({"epoch": epoch, "val_auc": val_auc}))
    return p


def test_no_checkpoints(tmp_path):
    ckpt = find_latest_checkpoint(tmp_path)
    assert ckpt is None


def test_finds_latest(tmp_path):
    _write_checkpoint(tmp_path, 3)
    _write_checkpoint(tmp_path, 7)
    _write_checkpoint(tmp_path, 5)
    ckpt = find_latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.epoch == 7


def test_checkpoint_is_valid(tmp_path):
    _write_checkpoint(tmp_path, 5)
    ckpt = find_latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.is_valid


def test_val_auc_parsed(tmp_path):
    _write_checkpoint(tmp_path, 5, val_auc=0.88)
    ckpt = find_latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert abs(ckpt.val_auc - 0.88) < 1e-6


def test_plan_start_fresh_no_ckpt(tmp_path):
    state = plan_resumption(tmp_path / "empty", total_epochs=20)
    assert state.flag == "START_FRESH"
    assert state.next_epoch == 0
    assert state.n_epochs_remaining == 20


def test_plan_resume(tmp_path):
    _write_checkpoint(tmp_path, 10)
    state = plan_resumption(tmp_path, total_epochs=20)
    assert state.flag == "RESUME"
    assert state.next_epoch == 11
    assert state.n_epochs_remaining == 9


def test_plan_complete(tmp_path):
    _write_checkpoint(tmp_path, 20)
    state = plan_resumption(tmp_path, total_epochs=20)
    assert state.flag == "COMPLETE"
    assert state.n_epochs_remaining == 0


def test_plan_invalid_total_epochs(tmp_path):
    state = plan_resumption(tmp_path, total_epochs=0)
    assert state.flag == "INVALID"


def test_format_start_fresh(tmp_path):
    state = plan_resumption(tmp_path / "empty", total_epochs=20)
    s = format_resumption_state(state)
    assert "START_FRESH" in s


def test_format_resume(tmp_path):
    _write_checkpoint(tmp_path, 5)
    state = plan_resumption(tmp_path, total_epochs=20)
    s = format_resumption_state(state)
    assert "RESUME" in s


def test_state_frozen(tmp_path):
    state = plan_resumption(tmp_path / "empty", total_epochs=20)
    try:
        state.flag = "MODIFIED"  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_custom_validate(tmp_path):
    # Write a file that passes a custom validator
    p = tmp_path / "ckpt_5.bin"
    p.write_bytes(b"\x00" * 100)
    ckpt = find_latest_checkpoint(tmp_path, validate_fn=lambda f: f.stat().st_size > 0)
    assert ckpt is not None
    assert ckpt.epoch == 5
