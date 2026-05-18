"""Deduplicate a list of pipeline candidates by period, epoch, and sky position.

Detects candidates that likely represent the same physical signal recovered
at different TIC IDs (due to crowding) or at harmonic periods.

Public API
----------
CandidatePair(idx_a, idx_b, tic_id_a, tic_id_b, period_ratio,
              sky_separation_arcsec, similarity_score, is_duplicate)
DeduplicationResult(n_input, n_unique, n_duplicates_removed, pairs,
                    unique_indices, flag)
deduplicate_candidates(candidates, *, period_tol_frac, epoch_tol_days,
                       sky_tol_arcsec, similarity_threshold) -> DeduplicationResult
format_deduplication_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePair:
    idx_a: int
    idx_b: int
    tic_id_a: int
    tic_id_b: int
    period_ratio: float
    sky_separation_arcsec: float
    similarity_score: float
    is_duplicate: bool


@dataclass(frozen=True)
class DeduplicationResult:
    n_input: int
    n_unique: int
    n_duplicates_removed: int
    pairs: tuple[CandidatePair, ...]
    unique_indices: tuple[int, ...]
    flag: str  # "OK" | "NO_DUPLICATES" | "EMPTY"


def _sky_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcseconds."""
    d2r = math.pi / 180.0
    cos_sep = (math.sin(dec1 * d2r) * math.sin(dec2 * d2r)
               + math.cos(dec1 * d2r) * math.cos(dec2 * d2r)
               * math.cos((ra1 - ra2) * d2r))
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep)) * 3600.0


def _period_similarity(pa: float, pb: float, tol_frac: float) -> tuple[float, float]:
    """Return (similarity_0_to_1, ratio) allowing integer harmonics 1:1 … 3:1."""
    if pa <= 0 or pb <= 0:
        return 0.0, 1.0
    best_sim = 0.0
    best_ratio = pa / pb
    for num in range(1, 4):
        for den in range(1, 4):
            ratio = num / den
            diff = abs(pa - pb * ratio) / max(pb * ratio, 1e-9)
            sim = max(0.0, 1.0 - diff / tol_frac)
            if sim > best_sim:
                best_sim = sim
                best_ratio = pa / pb
    return best_sim, best_ratio


def _get(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return default


def deduplicate_candidates(
    candidates: list[dict],
    *,
    period_tol_frac: float = 0.02,
    epoch_tol_days: float = 0.1,
    sky_tol_arcsec: float = 42.0,
    similarity_threshold: float = 0.85,
) -> DeduplicationResult:
    """Deduplicate candidates by period, epoch, and sky proximity.

    Args:
        candidates: List of candidate dicts with keys: tic_id, period_days,
            epoch_bjd, ra_deg, dec_deg, false_positive_probability (or fpp).
        period_tol_frac: Fractional period tolerance for harmonic matching.
        epoch_tol_days: Maximum epoch offset (days) for a match.
        sky_tol_arcsec: Maximum sky separation (arcsec) for a match.
        similarity_threshold: Combined similarity score above which a pair
            is marked as duplicate.

    Returns:
        :class:`DeduplicationResult`.
    """
    n = len(candidates)
    if n == 0:
        return DeduplicationResult(0, 0, 0, (), (), "EMPTY")

    pairs: list[CandidatePair] = []
    is_duplicate: list[bool] = [False] * n

    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = candidates[i], candidates[j]
            pa = _get(ca, "period_days", "best_period_days")
            pb = _get(cb, "period_days", "best_period_days")
            ea = _get(ca, "epoch_bjd", "epoch")
            eb = _get(cb, "epoch_bjd", "epoch")
            ra_a = _get(ca, "ra_deg", "ra")
            dec_a = _get(ca, "dec_deg", "dec")
            ra_b = _get(cb, "ra_deg", "ra")
            dec_b = _get(cb, "dec_deg", "dec")

            period_sim, ratio = _period_similarity(pa, pb, period_tol_frac)

            # Epoch similarity: mod by reference period
            ref_p = max(pa, pb, 1e-9)
            epoch_diff = abs((ea - eb) % ref_p)
            if epoch_diff > ref_p / 2:
                epoch_diff = ref_p - epoch_diff
            epoch_sim = max(0.0, 1.0 - epoch_diff / max(epoch_tol_days, 1e-9))

            # Sky separation
            sep_arcsec = _sky_sep_arcsec(ra_a, dec_a, ra_b, dec_b)
            sky_sim = max(0.0, 1.0 - sep_arcsec / max(sky_tol_arcsec, 1e-9))

            combined = (0.50 * period_sim + 0.30 * epoch_sim + 0.20 * sky_sim)
            dup = combined >= similarity_threshold

            pairs.append(CandidatePair(
                idx_a=i, idx_b=j,
                tic_id_a=int(_get(ca, "tic_id", default=0)),
                tic_id_b=int(_get(cb, "tic_id", default=0)),
                period_ratio=round(ratio, 6),
                sky_separation_arcsec=round(sep_arcsec, 2),
                similarity_score=round(combined, 4),
                is_duplicate=dup,
            ))

            if dup:
                # Keep the one with lower FPP (better candidate)
                fpp_a = _get(ca, "false_positive_probability", "fpp", "best_fpp", default=1.0)
                fpp_b = _get(cb, "false_positive_probability", "fpp", "best_fpp", default=1.0)
                if fpp_a <= fpp_b:
                    is_duplicate[j] = True
                else:
                    is_duplicate[i] = True

    unique_indices = tuple(i for i in range(n) if not is_duplicate[i])
    n_dup = sum(is_duplicate)
    flag = "NO_DUPLICATES" if n_dup == 0 else "OK"

    return DeduplicationResult(
        n_input=n,
        n_unique=len(unique_indices),
        n_duplicates_removed=n_dup,
        pairs=tuple(pairs),
        unique_indices=unique_indices,
        flag=flag,
    )


def format_deduplication_result(result: DeduplicationResult) -> str:
    """Format deduplication result as Markdown."""
    lines = [
        "## Candidate Deduplication",
        "",
        f"- Input candidates: {result.n_input}",
        f"- Unique candidates: {result.n_unique}",
        f"- Duplicates removed: {result.n_duplicates_removed}",
        f"- Pairs compared: {len(result.pairs)}",
        f"- **Flag: {result.flag}**",
    ]
    dup_pairs = [p for p in result.pairs if p.is_duplicate]
    if dup_pairs:
        lines += ["", "### Duplicate Pairs", ""]
        for p in dup_pairs[:10]:
            lines.append(
                f"- TIC {p.tic_id_a} ↔ TIC {p.tic_id_b}: "
                f"similarity={p.similarity_score:.3f}, "
                f"sep={p.sky_separation_arcsec:.1f}\""
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_deduplicator",
        description="Deduplicate pipeline candidates by period and sky position.",
    )
    parser.add_argument("--period-tol-frac", type=float, default=0.02)
    parser.add_argument("--sky-tol-arcsec", type=float, default=42.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    result = deduplicate_candidates(
        [],
        period_tol_frac=args.period_tol_frac,
        sky_tol_arcsec=args.sky_tol_arcsec,
        similarity_threshold=args.similarity_threshold,
    )
    print(format_deduplication_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
