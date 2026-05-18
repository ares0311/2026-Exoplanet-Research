"""Augment phase-folded light curve snippets for CNN training.

Applies data-augmentation transforms (noise injection, phase shifts,
flux scaling, time reversal) to increase training set diversity.

Public API
----------
AugmentedSnippet(original_tic_id, label, phase, flux, augmentation)
AugmentConfig(n_augmentations, noise_sigma, phase_shift_max,
              flux_scale_range, allow_reversal, seed)
augment_snippet(snippet, config) -> list[AugmentedSnippet]
augment_dataset(snippets, config) -> list[AugmentedSnippet]
format_augmentation_summary(originals, augmented) -> str
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentConfig:
    n_augmentations: int = 4
    noise_sigma: float = 2e-4      # fractional noise level (of flux ~1.0)
    phase_shift_max: float = 0.05  # max phase shift as fraction of period
    flux_scale_range: tuple[float, float] = (0.98, 1.02)
    allow_reversal: bool = True
    seed: int = 42


@dataclass(frozen=True)
class AugmentedSnippet:
    original_tic_id: int
    label: int
    phase: tuple[float, ...]
    flux: tuple[float, ...]
    augmentation: str      # description of transform applied


def _roll(lst: list[float], shift: int) -> list[float]:
    n = len(lst)
    if n == 0:
        return lst
    shift = shift % n
    return lst[shift:] + lst[:shift]


def augment_snippet(
    snippet: LabelledSnippet,  # type: ignore[name-defined]  # noqa: F821
    config: AugmentConfig,
) -> list[AugmentedSnippet]:
    """Generate augmented copies of a single snippet.

    Args:
        snippet: A :class:`~labelled_lc_collector.LabelledSnippet`.
        config: Augmentation configuration.

    Returns:
        List of :class:`AugmentedSnippet` objects.
    """
    rng = random.Random(config.seed ^ snippet.tic_id)
    results: list[AugmentedSnippet] = []
    flux_list = list(snippet.flux)
    n = len(flux_list)

    for _i in range(config.n_augmentations):
        f = flux_list[:]
        transforms: list[str] = []

        # Gaussian noise
        if config.noise_sigma > 0:
            f = [v + rng.gauss(0, config.noise_sigma) for v in f]
            transforms.append("noise")

        # Phase shift (cyclic roll)
        if config.phase_shift_max > 0:
            max_shift_bins = int(config.phase_shift_max * n)
            shift = rng.randint(-max_shift_bins, max_shift_bins)
            if shift != 0:
                f = _roll(f, shift)
                transforms.append(f"shift{shift:+d}")

        # Flux scaling
        lo, hi = config.flux_scale_range
        scale = rng.uniform(lo, hi)
        if abs(scale - 1.0) > 1e-6:
            f = [v * scale for v in f]
            transforms.append(f"scale{scale:.4f}")

        # Time reversal
        if config.allow_reversal and rng.random() < 0.3:
            f = f[::-1]
            transforms.append("reverse")

        aug_label = "+".join(transforms) if transforms else "identity"

        results.append(AugmentedSnippet(
            original_tic_id=snippet.tic_id,
            label=snippet.label,
            phase=snippet.phase,
            flux=tuple(round(v, 8) for v in f),
            augmentation=aug_label,
        ))

    return results


def augment_dataset(
    snippets: list[LabelledSnippet],  # type: ignore[name-defined]  # noqa: F821
    config: AugmentConfig | None = None,
) -> list[AugmentedSnippet]:
    """Augment all snippets in a dataset.

    Args:
        snippets: List of :class:`~labelled_lc_collector.LabelledSnippet`.
        config: Augmentation config; uses defaults if None.

    Returns:
        List of all augmented snippets (originals not included).
    """
    if config is None:
        config = AugmentConfig()
    result: list[AugmentedSnippet] = []
    for snip in snippets:
        result.extend(augment_snippet(snip, config))
    return result


def format_augmentation_summary(
    n_originals: int,
    augmented: list[AugmentedSnippet],
) -> str:
    """Format augmentation summary as Markdown."""
    n_aug = len(augmented)
    label_counts: dict[int, int] = {}
    for a in augmented:
        label_counts[a.label] = label_counts.get(a.label, 0) + 1

    lines = [
        "## CNN Feature Augmentation Summary",
        "",
        f"- Original snippets: {n_originals}",
        f"- Augmented snippets generated: {n_aug}",
        f"- Positive augmented: {label_counts.get(1, 0)}",
        f"- Negative augmented: {label_counts.get(0, 0)}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="cnn_feature_augmenter",
        description="Augment phase-folded LC snippets for CNN training.",
    )
    parser.add_argument("--dataset", required=True, metavar="JSON")
    parser.add_argument("--n-aug", type=int, default=4)
    parser.add_argument("--noise-sigma", type=float, default=2e-4)
    parser.add_argument("--output", required=True, metavar="JSON")
    args = parser.parse_args(argv)

    # Lazy import to avoid circular dependency when running as script
    from Skills.labelled_lc_collector import LabelledSnippet

    raw = _json.loads(_Path(args.dataset).read_text())
    snippets = []
    for s in raw.get("snippets", []):
        snippets.append(LabelledSnippet(
            tic_id=s["tic_id"],
            label=s["label"],
            period_days=s["period_days"],
            epoch_bjd=s["epoch_bjd"],
            phase=tuple(s["phase"]),
            flux=tuple(s["flux"]),
            n_points=s["n_points"],
            source=s["source"],
        ))

    config = AugmentConfig(n_augmentations=args.n_aug, noise_sigma=args.noise_sigma)
    augmented = augment_dataset(snippets, config)

    out_data = [
        {"tic_id": a.original_tic_id, "label": a.label,
         "phase": list(a.phase), "flux": list(a.flux),
         "augmentation": a.augmentation}
        for a in augmented
    ]
    _Path(args.output).write_text(_json.dumps(out_data, indent=2))
    print(format_augmentation_summary(len(snippets), augmented))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
