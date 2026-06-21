"""Tests for Skills/build_joint_cnn_splits.py."""
from __future__ import annotations

import json
import random
from pathlib import Path

from Skills.build_joint_cnn_splits import build_joint_splits, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_split_dir(
    tmp_path: Path,
    name: str,
    n_train: int = 20,
    n_val: int = 5,
    n_test: int = 5,
) -> Path:
    d = tmp_path / name
    d.mkdir()
    rng = random.Random(42)

    def _examples(n: int, split: str) -> dict:
        examples = [
            {"flux": [float(i % 3)] * 201, "label": i % 2, "source": name}
            for i in range(n)
        ]
        rng.shuffle(examples)
        return {"split": split, "examples": examples}

    (d / "train.json").write_text(json.dumps(_examples(n_train, "train")))
    (d / "val.json").write_text(json.dumps(_examples(n_val, "val")))
    (d / "test.json").write_text(json.dumps(_examples(n_test, "test")))
    return d


# ---------------------------------------------------------------------------
# build_joint_splits
# ---------------------------------------------------------------------------


class TestBuildJointSplits:
    def test_joint_train_count(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_train=20)
        kepler = _make_split_dir(tmp_path, "kepler", n_train=10)
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out)
        assert manifest["joint_train_n"] == 30

    def test_tess_val_and_test_unchanged(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_val=6, n_test=7)
        kepler = _make_split_dir(tmp_path, "kepler", n_val=99)
        out = tmp_path / "joint"
        build_joint_splits(tess, kepler, out)
        val = json.loads((out / "val.json").read_text())
        test = json.loads((out / "test.json").read_text())
        tess_val = json.loads((tess / "val.json").read_text())
        tess_test = json.loads((tess / "test.json").read_text())
        assert val == tess_val
        assert test == tess_test

    def test_manifest_written(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out = tmp_path / "joint"
        build_joint_splits(tess, kepler, out)
        manifest = json.loads((out / "manifest.json").read_text())
        assert "joint_train_n" in manifest
        assert "tess_train_n" in manifest
        assert "kepler_train_n" in manifest

    def test_manifest_counts_match_files(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_train=12, n_val=4, n_test=4)
        kepler = _make_split_dir(tmp_path, "kepler", n_train=8)
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out)
        train = json.loads((out / "train.json").read_text())
        examples = train if isinstance(train, list) else train["examples"]
        assert len(examples) == manifest["joint_train_n"]

    def test_max_kepler_cap(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_train=10)
        kepler = _make_split_dir(tmp_path, "kepler", n_train=20)
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out, max_kepler=5)
        assert manifest["kepler_train_n"] == 5
        assert manifest["joint_train_n"] == 15

    def test_max_kepler_larger_than_available(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_train=10)
        kepler = _make_split_dir(tmp_path, "kepler", n_train=8)
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out, max_kepler=100)
        assert manifest["kepler_train_n"] == 8

    def test_output_dir_created(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out = tmp_path / "nested" / "joint"
        build_joint_splits(tess, kepler, out)
        assert out.is_dir()
        assert (out / "train.json").exists()

    def test_seed_determinism(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out_a = tmp_path / "joint_a"
        out_b = tmp_path / "joint_b"
        build_joint_splits(tess, kepler, out_a, seed=42)
        build_joint_splits(tess, kepler, out_b, seed=42)
        train_a = (out_a / "train.json").read_text()
        train_b = (out_b / "train.json").read_text()
        assert train_a == train_b

    def test_different_seeds_produce_different_order(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess", n_train=30)
        kepler = _make_split_dir(tmp_path, "kepler", n_train=20)
        out_a = tmp_path / "joint_a"
        out_b = tmp_path / "joint_b"
        build_joint_splits(tess, kepler, out_a, seed=7)
        build_joint_splits(tess, kepler, out_b, seed=99)
        train_a = (out_a / "train.json").read_text()
        train_b = (out_b / "train.json").read_text()
        assert train_a != train_b

    def test_label_counts_in_manifest(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out)
        assert manifest["joint_train_positive"] + manifest["joint_train_negative"] == manifest[
            "joint_train_n"
        ]

    def test_list_format_split_files(self, tmp_path: Path) -> None:
        """Accept raw list format (no 'split'/'examples' wrapper)."""
        tess = tmp_path / "tess"
        tess.mkdir()
        examples = [{"flux": [1.0] * 201, "label": i % 2} for i in range(10)]
        for split in ("train", "val", "test"):
            (tess / f"{split}.json").write_text(json.dumps(examples[:5]))
        kepler = _make_split_dir(tmp_path, "kepler", n_train=5)
        out = tmp_path / "joint"
        manifest = build_joint_splits(tess, kepler, out)
        assert manifest["joint_train_n"] == 10

    def test_return_value_is_dict(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out = tmp_path / "joint"
        result = build_joint_splits(tess, kepler, out)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_success(self, tmp_path: Path) -> None:
        tess = _make_split_dir(tmp_path, "tess")
        kepler = _make_split_dir(tmp_path, "kepler")
        out = tmp_path / "joint"
        rc = main(
            [
                "--tess-split-dir",
                str(tess),
                "--kepler-split-dir",
                str(kepler),
                "--output-dir",
                str(out),
            ]
        )
        assert rc == 0
        assert (out / "train.json").exists()

    def test_main_missing_tess_dir_returns_1(self, tmp_path: Path) -> None:
        kepler = _make_split_dir(tmp_path, "kepler")
        rc = main(
            [
                "--tess-split-dir",
                str(tmp_path / "nonexistent"),
                "--kepler-split-dir",
                str(kepler),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1
