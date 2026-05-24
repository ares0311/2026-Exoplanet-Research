"""Format a scored candidate for submission to TFOP or Planet Hunters.

Generates a structured plain-text submission record from an ``exo`` pipeline
output dict, mapping pipeline keys to the fields expected by the TFOP Working
Group and Planet Hunters TESS submission templates.  Does not perform any
scoring — it is a pure formatter.  Distinct from ``export_candidates`` (CSV/Markdown
table) and ``candidate_report_card`` (full vetting card).

Public API
----------
SubmissionRecord(tic_id, pathway, period_days, epoch_bjd, depth_ppm,
                 duration_hours, planet_radius_rearth, fpp,
                 detection_confidence, transit_count, sectors,
                 stellar_radius_rsun, stellar_mass_msun, stellar_teff_k,
                 notes, template, formatted_text)
format_submission(candidate_dict, *, template, extra_notes) -> SubmissionRecord
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TFOP_TEMPLATE = "tfop_wg"
_PH_TEMPLATE = "planet_hunters"
_VALID_TEMPLATES = {_TFOP_TEMPLATE, _PH_TEMPLATE}


@dataclass(frozen=True)
class SubmissionRecord:
    tic_id: int | None
    pathway: str | None
    period_days: float | None
    epoch_bjd: float | None
    depth_ppm: float | None
    duration_hours: float | None
    planet_radius_rearth: float | None
    fpp: float | None
    detection_confidence: float | None
    transit_count: int | None
    sectors: str | None          # comma-separated list or free text
    stellar_radius_rsun: float | None
    stellar_mass_msun: float | None
    stellar_teff_k: float | None
    notes: str
    template: str
    formatted_text: str
    flag: str  # "OK" | "INVALID" | "INCOMPLETE"


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup with fallback."""
    for key in keys:
        if key in d:
            return d[key]
    return default


def _fmt(val: Any, fmt: str = "") -> str:
    if val is None:
        return "N/A"
    if fmt:
        return format(val, fmt)
    return str(val)


def format_submission(
    candidate_dict: dict[str, Any],
    *,
    template: str = _TFOP_TEMPLATE,
    extra_notes: str = "",
) -> SubmissionRecord:
    """Build a structured submission record from a pipeline output dict.

    Args:
        candidate_dict: Output dict from ``run_pipeline`` / ``exo`` CLI.
        template: ``"tfop_wg"`` or ``"planet_hunters"``.
        extra_notes: Free-text notes appended to the submission.

    Returns:
        :class:`SubmissionRecord`.
    """
    if not isinstance(candidate_dict, dict):
        return SubmissionRecord(
            None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, extra_notes, template, "", "INVALID"
        )
    if template not in _VALID_TEMPLATES:
        return SubmissionRecord(
            None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, extra_notes, template, "", "INVALID"
        )

    # Extract fields — support both flat and nested pipeline output formats
    tic_id = _get(candidate_dict, "tic_id", "TIC_ID")
    if tic_id is not None:
        try:
            tic_id = int(tic_id)
        except (TypeError, ValueError):
            tic_id = None

    period = _get(candidate_dict, "period_days", "period")
    epoch = _get(candidate_dict, "epoch_bjd", "epoch", "t0_bjd")
    depth = _get(candidate_dict, "depth_ppm", "depth")
    duration = _get(candidate_dict, "duration_hours")
    rp = _get(candidate_dict, "planet_radius_rearth", "radius_rearth")
    fpp = _get(candidate_dict, "false_positive_probability", "fpp")
    if fpp is None:
        scores = candidate_dict.get("scores", {})
        if isinstance(scores, dict):
            fpp = scores.get("false_positive_probability")
    dc = _get(candidate_dict, "detection_confidence")
    if dc is None:
        scores = candidate_dict.get("scores", {})
        if isinstance(scores, dict):
            dc = scores.get("detection_confidence")
    n_transit = _get(candidate_dict, "transit_count", "n_transits")
    sectors = _get(candidate_dict, "sectors")
    if isinstance(sectors, (list, tuple)):
        sectors = ", ".join(str(s) for s in sectors)

    sr = _get(candidate_dict, "stellar_radius_rsun", "stellar_radius")
    sm = _get(candidate_dict, "stellar_mass_msun", "stellar_mass")
    st = _get(candidate_dict, "stellar_teff_k", "teff_k")
    pathway = _get(candidate_dict, "pathway", "submission_pathway")

    # Check for minimum required fields
    missing = [k for k, v in (("TIC ID", tic_id), ("period", period),
                               ("epoch", epoch), ("depth", depth)) if v is None]
    flag = "INCOMPLETE" if missing else "OK"
    notes_full = extra_notes
    if missing:
        notes_full = f"Missing fields: {', '.join(missing)}. " + notes_full

    # Build formatted text
    if template == _TFOP_TEMPLATE:
        formatted_text = _build_tfop_text(
            tic_id, period, epoch, depth, duration, rp, fpp, dc,
            n_transit, sectors, sr, sm, st, pathway, notes_full,
        )
    else:
        formatted_text = _build_ph_text(
            tic_id, period, epoch, depth, duration, rp, fpp,
            n_transit, sectors, notes_full,
        )

    return SubmissionRecord(
        tic_id=tic_id,
        pathway=pathway,
        period_days=period,
        epoch_bjd=epoch,
        depth_ppm=depth,
        duration_hours=duration,
        planet_radius_rearth=rp,
        fpp=fpp,
        detection_confidence=dc,
        transit_count=n_transit,
        sectors=sectors,
        stellar_radius_rsun=sr,
        stellar_mass_msun=sm,
        stellar_teff_k=st,
        notes=notes_full,
        template=template,
        formatted_text=formatted_text,
        flag=flag,
    )


def _build_tfop_text(tic_id, period, epoch, depth, duration, rp,
                     fpp, dc, n_transit, sectors, sr, sm, st, pathway, notes):
    lines = [
        "=== TFOP Working Group Submission ===",
        "",
        f"TIC ID          : {_fmt(tic_id)}",
        f"Pathway         : {_fmt(pathway)}",
        f"Period (days)   : {_fmt(period, '.6f')}",
        f"Epoch (BJD-TDB) : {_fmt(epoch, '.6f')}",
        f"Depth (ppm)     : {_fmt(depth, '.1f')}",
        f"Duration (hours): {_fmt(duration, '.3f')}",
        f"Planet Rp (R⊕)  : {_fmt(rp, '.2f')}",
        f"FPP             : {_fmt(fpp, '.4f')}",
        f"Detection conf. : {_fmt(dc, '.4f')}",
        f"N transits      : {_fmt(n_transit)}",
        f"Sectors         : {_fmt(sectors)}",
        "",
        "--- Stellar Parameters ---",
        f"Radius (R☉)     : {_fmt(sr, '.3f')}",
        f"Mass (M☉)       : {_fmt(sm, '.3f')}",
        f"Teff (K)        : {_fmt(st, '.0f')}",
        "",
        "--- Notes ---",
        notes or "(none)",
        "",
        "DISCLAIMER: candidate signal only; confirmation requires follow-up observations.",
    ]
    return "\n".join(lines)


def _build_ph_text(tic_id, period, epoch, depth, duration, rp, fpp, n_transit, sectors, notes):
    lines = [
        "=== Planet Hunters TESS Submission ===",
        "",
        f"TIC ID       : {_fmt(tic_id)}",
        f"Period (d)   : {_fmt(period, '.6f')}",
        f"Epoch (BJD)  : {_fmt(epoch, '.6f')}",
        f"Depth (ppm)  : {_fmt(depth, '.1f')}",
        f"Duration (h) : {_fmt(duration, '.3f')}",
        f"Rp (R⊕)      : {_fmt(rp, '.2f')}",
        f"FPP          : {_fmt(fpp, '.4f')}",
        f"N transits   : {_fmt(n_transit)}",
        f"TESS sectors : {_fmt(sectors)}",
        "",
        "Notes:",
        notes or "(none)",
        "",
        "DISCLAIMER: candidate signal only — not a confirmed planet.",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        prog="candidate_submission_formatter",
        description="Format a pipeline candidate dict for TFOP/Planet Hunters submission.",
    )
    parser.add_argument("--template", choices=list(_VALID_TEMPLATES), default=_TFOP_TEMPLATE)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON string of candidate dict (for testing)")
    args = parser.parse_args(argv)

    if args.json:
        cand = _json.loads(args.json)
    else:
        cand = {"tic_id": 0, "period_days": None, "epoch_bjd": None, "depth_ppm": None}

    rec = format_submission(cand, template=args.template, extra_notes=args.notes)
    print(rec.formatted_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
