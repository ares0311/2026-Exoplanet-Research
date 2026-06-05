"""Classify warm Jupiter formation/migration pathway from orbital architecture."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WarmJupiterMigrationResult:
    migration_class: str        # DISK_DRIVEN / HIGH_E / IN_SITU / AMBIGUOUS
    disk_score: float           # [0, 1] probability-like score for disk migration
    high_e_score: float         # [0, 1] score for high-e migration
    in_situ_score: float        # [0, 1] score for in-situ formation
    evidence: tuple[str, ...]   # brief reasons
    flag: str


def classify_warm_jupiter(
    period_days: float,
    eccentricity: float = 0.0,
    has_nearby_companion: bool = False,
    companion_period_days: float | None = None,
    companion_mass_mjup: float | None = None,
    planet_mass_mjup: float = 1.0,
) -> WarmJupiterMigrationResult:
    """Classify likely formation/migration for a warm Jupiter.

    Based on Dawson & Johnson (2018) warm Jupiter review:
      - Disk-driven: circular (e < 0.1), compact multi, outer companion
      - High-e: eccentric (e > 0.2), no nearby companions, wide outer companion
      - In-situ: very short period (P < 5d), circular, no companion signature

    Args:
        period_days: orbital period (days)
        eccentricity: eccentricity (0 = circular)
        has_nearby_companion: inner/outer companion present within 3:1 MMR
        companion_period_days: outer companion period (days), if known
        companion_mass_mjup: outer companion mass (Jupiter masses), if known
        planet_mass_mjup: planet mass (Jupiter masses)
    """
    if period_days <= 0.0:
        return WarmJupiterMigrationResult("UNKNOWN", 0.0, 0.0, 0.0, (), "INVALID_PERIOD")
    if not (0.0 <= eccentricity < 1.0):
        return WarmJupiterMigrationResult("UNKNOWN", 0.0, 0.0, 0.0, (), "INVALID_ECCENTRICITY")

    evidence: list[str] = []
    disk = 0.0
    high_e = 0.0
    in_situ = 0.0

    # Period range check (warm Jupiter: 10-200 d; hot Jupiter < 10d)
    is_hot = period_days < 10.0
    is_warm = 10.0 <= period_days <= 200.0

    if is_hot:
        evidence.append("period < 10 d: hot Jupiter territory")
        disk += 0.3
        in_situ += 0.3
    elif is_warm:
        evidence.append(f"period {period_days:.1f} d: warm Jupiter regime")
        disk += 0.2

    # Eccentricity evidence
    if eccentricity < 0.05:
        disk += 0.35
        in_situ += 0.25
        evidence.append("circular orbit supports disk migration or in-situ")
    elif eccentricity < 0.2:
        disk += 0.15
        high_e += 0.15
        evidence.append("moderate eccentricity ambiguous")
    else:
        high_e += 0.4
        evidence.append(f"eccentricity {eccentricity:.2f} favours high-e migration")

    # Companion evidence
    if has_nearby_companion:
        disk += 0.3
        in_situ += 0.1
        evidence.append("nearby companion supports disk migration or in-situ")
    else:
        high_e += 0.2
        evidence.append("no nearby companion consistent with high-e scattering")

    if companion_period_days is not None and companion_period_days > period_days * 3:
        if companion_mass_mjup is not None and companion_mass_mjup > 0.5:
            high_e += 0.2
            disk += 0.1
            evidence.append("massive outer companion (Kozai or secular excitation possible)")
        else:
            disk += 0.1
            evidence.append("outer companion present")

    # Very short period: in-situ bonus
    if period_days < 5.0 and eccentricity < 0.05:
        in_situ += 0.3
        evidence.append("very short period + circular: in-situ signature")

    # Normalise scores
    total = disk + high_e + in_situ
    if total > 0:
        disk /= total
        high_e /= total
        in_situ /= total

    best = max(disk, high_e, in_situ)
    if best < 0.45:
        mclass = "AMBIGUOUS"
    elif disk == best:
        mclass = "DISK_DRIVEN"
    elif high_e == best:
        mclass = "HIGH_E"
    else:
        mclass = "IN_SITU"

    return WarmJupiterMigrationResult(
        migration_class=mclass,
        disk_score=disk,
        high_e_score=high_e,
        in_situ_score=in_situ,
        evidence=tuple(evidence),
        flag="OK",
    )


def format_warm_jupiter_result(r: WarmJupiterMigrationResult) -> str:
    if r.flag != "OK":
        return f"WarmJupiterMigration | flag={r.flag}"
    ev = "; ".join(r.evidence) if r.evidence else "none"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Migration class | {r.migration_class} |\n"
        f"| Disk-driven score | {r.disk_score:.2f} |\n"
        f"| High-e score | {r.high_e_score:.2f} |\n"
        f"| In-situ score | {r.in_situ_score:.2f} |\n"
        f"| Evidence | {ev} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Warm Jupiter migration classifier")
    p.add_argument("period_days", type=float)
    p.add_argument("--ecc", type=float, default=0.0)
    p.add_argument("--companion", action="store_true")
    args = p.parse_args()
    r = classify_warm_jupiter(args.period_days, eccentricity=args.ecc,
                               has_nearby_companion=args.companion)
    print(format_warm_jupiter_result(r))


if __name__ == "__main__":
    _cli()
