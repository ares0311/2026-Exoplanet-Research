"""Tests for Skills.cnn_feature_augmenter."""
from __future__ import annotations

from Skills.cnn_feature_augmenter import (
    AugmentConfig,
    AugmentedSnippet,
    augment_dataset,
    augment_snippet,
    format_augmentation_summary,
)
from Skills.labelled_lc_collector import LabelledSnippet


def _make_snippet(tic_id=1, label=1, n_bins=51):
    phase = tuple((-0.5 + (i + 0.5) / n_bins) for i in range(n_bins))
    # Simple transit-like flux
    flux = tuple(
        1.0 - 0.01 if abs(p) < 0.05 else 1.0 for p in phase
    )
    return LabelledSnippet(
        tic_id=tic_id,
        label=label,
        period_days=5.0,
        epoch_bjd=2458000.0,
        phase=phase,
        flux=flux,
        n_points=500,
        source="tess",
    )


class TestAugmentSnippet:
    def test_returns_list(self) -> None:
        snip = _make_snippet()
        config = AugmentConfig(n_augmentations=3)
        result = augment_snippet(snip, config)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_is_augmented_snippet(self) -> None:
        snip = _make_snippet()
        result = augment_snippet(snip, AugmentConfig())
        for a in result:
            assert isinstance(a, AugmentedSnippet)

    def test_label_preserved(self) -> None:
        snip = _make_snippet(label=0)
        result = augment_snippet(snip, AugmentConfig())
        for a in result:
            assert a.label == 0

    def test_original_tic_id_preserved(self) -> None:
        snip = _make_snippet(tic_id=42)
        result = augment_snippet(snip, AugmentConfig())
        for a in result:
            assert a.original_tic_id == 42

    def test_flux_length_preserved(self) -> None:
        snip = _make_snippet()
        result = augment_snippet(snip, AugmentConfig())
        for a in result:
            assert len(a.flux) == len(snip.flux)

    def test_augmentation_label_set(self) -> None:
        snip = _make_snippet()
        result = augment_snippet(snip, AugmentConfig())
        for a in result:
            assert len(a.augmentation) > 0

    def test_no_augmentation_returns_empty_on_zero(self) -> None:
        snip = _make_snippet()
        result = augment_snippet(snip, AugmentConfig(n_augmentations=0))
        assert result == []


class TestAugmentDataset:
    def test_returns_list(self) -> None:
        snippets = [_make_snippet(tic_id=i) for i in range(3)]
        result = augment_dataset(snippets)
        assert isinstance(result, list)

    def test_n_augmented_correct(self) -> None:
        snippets = [_make_snippet(tic_id=i) for i in range(3)]
        config = AugmentConfig(n_augmentations=2)
        result = augment_dataset(snippets, config)
        assert len(result) == 6

    def test_empty_input(self) -> None:
        result = augment_dataset([])
        assert result == []

    def test_default_config_used(self) -> None:
        snippets = [_make_snippet()]
        result = augment_dataset(snippets, None)
        assert len(result) == AugmentConfig().n_augmentations


class TestFormatAugmentationSummary:
    def test_returns_string(self) -> None:
        snippets = [_make_snippet()]
        result = augment_dataset(snippets)
        assert isinstance(format_augmentation_summary(1, result), str)

    def test_contains_counts(self) -> None:
        snippets = [_make_snippet(tic_id=i) for i in range(2)]
        result = augment_dataset(snippets, AugmentConfig(n_augmentations=3))
        out = format_augmentation_summary(len(snippets), result)
        assert "6" in out
