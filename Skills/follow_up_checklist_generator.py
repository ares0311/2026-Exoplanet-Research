"""Generate observation checklists for follow-up of transit candidates.

Public API
----------
ChecklistItem(step, description, priority, done)
FollowUpChecklist(tic_id, period_days, checklist, n_total, n_high, flag)
generate_checklist(tic_id, *, period_days, fpp, pathway, stellar_teff_k,
                   n_transits) -> FollowUpChecklist
format_checklist(checklist) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistItem:
    step: int
    description: str
    priority: str   # "HIGH" | "MEDIUM" | "LOW"
    done: bool


@dataclass(frozen=True)
class FollowUpChecklist:
    tic_id: int | None
    period_days: float | None
    checklist: tuple[ChecklistItem, ...]
    n_total: int
    n_high: int
    flag: str  # "OK" | "MINIMAL" | "EMPTY"


def _add(items: list[ChecklistItem], step: int, desc: str,
         priority: str = "MEDIUM") -> int:
    items.append(ChecklistItem(step=step, description=desc,
                               priority=priority, done=False))
    return step + 1


def generate_checklist(
    tic_id: int | None = None,
    *,
    period_days: float | None = None,
    fpp: float | None = None,
    pathway: str | None = None,
    stellar_teff_k: float | None = None,
    n_transits: int | None = None,
) -> FollowUpChecklist:
    """Generate a follow-up checklist for a transit candidate.

    Args:
        tic_id: TIC identifier.
        period_days: Orbital period.
        fpp: False-positive probability.
        pathway: Submission pathway string.
        stellar_teff_k: Stellar effective temperature in K.
        n_transits: Number of observed transits.

    Returns:
        FollowUpChecklist with prioritized steps.
    """
    items: list[ChecklistItem] = []
    step = 1

    # Always check
    step = _add(items, step, "Verify transit depth consistency across all sectors", "HIGH")
    step = _add(items, step, "Check odd-even transit depth ratio (EB discriminator)", "HIGH")
    step = _add(items, step, "Inspect centroid shift during transit", "HIGH")
    step = _add(items, step, "Check for secondary eclipse at phase 0.5", "HIGH")

    if n_transits is not None and n_transits < 3:
        step = _add(items, step,
                    f"Only {n_transits} transit(s) observed — schedule additional observations",
                    "HIGH")

    if fpp is not None and fpp > 0.10:
        step = _add(items, step,
                    f"FPP={fpp:.2f} is elevated — obtain spectroscopic stellar parameters",
                    "HIGH")

    # Stellar context
    if stellar_teff_k is not None:
        if stellar_teff_k < 4000:
            step = _add(items, step, "M-dwarf host — check for stellar flares in light curve",
                        "HIGH")
        elif stellar_teff_k > 7000:
            step = _add(items, step, "Hot star — check for pulsations or ellipsoidal variations",
                        "MEDIUM")

    # Period-dependent checks
    if period_days is not None:
        if period_days < 2.0:
            step = _add(items, step,
                        "Ultra-short period — check for tidal circularisation and phase-curve",
                        "MEDIUM")
        if period_days > 10.0:
            step = _add(items, step,
                        "Long period — accumulate additional TESS sectors if available",
                        "MEDIUM")

    # Pathway-specific
    if pathway == "tfop_ready":
        step = _add(items, step, "Submit to TFOP Working Group for ground-based follow-up",
                    "HIGH")
        step = _add(items, step, "Plan ground-based photometric transit observation", "HIGH")
        step = _add(items, step, "Request reconnaissance spectroscopy (e.g. CHIRON/TRES)", "HIGH")
    elif pathway == "planet_hunters_discussion":
        step = _add(items, step, "Post to Planet Hunters TESS Talk for community review",
                    "MEDIUM")

    # Standard medium/low priority
    step = _add(items, step, "Cross-match TIC with Gaia DR3 for neighbour contamination",
                "MEDIUM")
    step = _add(items, step, "Check NEA and ExoFOP for prior follow-up observations", "MEDIUM")
    step = _add(items, step, "Run injection-recovery to estimate detection completeness", "LOW")
    step = _add(items, step, "Archive processed light curve and pipeline outputs", "LOW")

    n_total = len(items)
    n_high = sum(1 for i in items if i.priority == "HIGH")
    flag = "OK" if n_total >= 4 else "MINIMAL"

    return FollowUpChecklist(
        tic_id=tic_id,
        period_days=period_days,
        checklist=tuple(items),
        n_total=n_total,
        n_high=n_high,
        flag=flag,
    )


def format_checklist(checklist: FollowUpChecklist) -> str:
    """Format a follow-up checklist as Markdown.

    Args:
        checklist: FollowUpChecklist to format.

    Returns:
        Markdown string with checkbox items.
    """
    tic_str = str(checklist.tic_id) if checklist.tic_id is not None else "Unknown"
    period_str = (f"{checklist.period_days:.4f} d"
                  if checklist.period_days is not None else "Unknown")
    lines = [
        f"## Follow-up Checklist — TIC {tic_str}\n",
        f"**Period**: {period_str} | "
        f"**Items**: {checklist.n_total} ({checklist.n_high} high-priority)\n",
        "",
    ]
    for item in checklist.checklist:
        box = "[ ]" if not item.done else "[x]"
        pri_tag = f"[**{item.priority}**]" if item.priority == "HIGH" else f"[{item.priority}]"
        lines.append(f"- {box} {pri_tag} {item.description}")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate follow-up checklist.")
    parser.add_argument("--tic-id", type=int, default=None)
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--fpp", type=float, default=None)
    parser.add_argument("--pathway", default=None)
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--n-transits", type=int, default=None)
    args = parser.parse_args(argv)

    cl = generate_checklist(
        args.tic_id,
        period_days=args.period,
        fpp=args.fpp,
        pathway=args.pathway,
        stellar_teff_k=args.teff,
        n_transits=args.n_transits,
    )
    print(format_checklist(cl))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
