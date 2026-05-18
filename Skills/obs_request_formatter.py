"""Format ground-based follow-up observation requests for candidates.

Generates standardised observation request documents (Markdown and JSON)
suitable for submitting to TFOP, PEST, or other ground-based networks.

Public API
----------
ObsRequest(tic_id, ra_deg, dec_deg, tmag, period_days, epoch_bjd,
           duration_hours, depth_ppm, priority, notes, contact_email)
ObsRequestResult(request, markdown, json_payload)
build_obs_request(tic_id, ra_deg, dec_deg, tmag, period_days, epoch_bjd,
                  duration_hours, depth_ppm, *, priority, notes,
                  contact_email) -> ObsRequestResult
format_obs_request(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ObsRequest:
    tic_id: int
    ra_deg: float
    dec_deg: float
    tmag: float
    period_days: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float
    priority: str               # "HIGH", "MEDIUM", "LOW"
    notes: str
    contact_email: str


@dataclass(frozen=True)
class ObsRequestResult:
    request: ObsRequest
    markdown: str
    json_payload: str


def _ra_to_hms(ra_deg: float) -> str:
    """Convert RA degrees to HH:MM:SS.s string."""
    ra_h = ra_deg / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = ((ra_h - h) * 60 - m) * 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _dec_to_dms(dec_deg: float) -> str:
    """Convert Dec degrees to ±DD:MM:SS string."""
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    deg = int(d)
    m = int((d - deg) * 60)
    s = ((d - deg) * 60 - m) * 60
    return f"{sign}{deg:02d}:{m:02d}:{s:02.0f}"


def build_obs_request(
    tic_id: int,
    ra_deg: float,
    dec_deg: float,
    tmag: float,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    depth_ppm: float,
    *,
    priority: str = "MEDIUM",
    notes: str = "",
    contact_email: str = "",
) -> ObsRequestResult:
    """Build a ground-based follow-up observation request.

    Args:
        tic_id: TESS Input Catalog ID.
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.
        tmag: TESS magnitude.
        period_days: Orbital period in days.
        epoch_bjd: Reference transit epoch (BJD).
        duration_hours: Transit duration in hours.
        depth_ppm: Transit depth in ppm.
        priority: Observation priority ("HIGH", "MEDIUM", "LOW").
        notes: Free-text notes.
        contact_email: Contact email for the request.

    Returns:
        :class:`ObsRequestResult`.
    """
    valid_priorities = {"HIGH", "MEDIUM", "LOW"}
    priority = (
        "MEDIUM" if priority.upper() not in valid_priorities else priority.upper()
    )

    req = ObsRequest(
        tic_id=int(tic_id),
        ra_deg=float(ra_deg),
        dec_deg=float(dec_deg),
        tmag=float(tmag),
        period_days=float(period_days),
        epoch_bjd=float(epoch_bjd),
        duration_hours=float(duration_hours),
        depth_ppm=float(depth_ppm),
        priority=priority,
        notes=str(notes),
        contact_email=str(contact_email),
    )

    ra_hms = _ra_to_hms(ra_deg)
    dec_dms = _dec_to_dms(dec_deg)
    depth_pct = depth_ppm / 1e4

    md_lines = [
        f"# Follow-Up Observation Request: TIC {tic_id}",
        "",
        f"**Priority:** {priority}",
        "",
        "## Target",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| TIC ID | {tic_id} |",
        f"| RA | {ra_deg:.6f}° ({ra_hms}) |",
        f"| Dec | {dec_deg:.6f}° ({dec_dms}) |",
        f"| Tmag | {tmag:.2f} |",
        "",
        "## Transit Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Period | {period_days:.6f} d |",
        f"| Epoch (BJD) | {epoch_bjd:.4f} |",
        f"| Duration | {duration_hours:.2f} h |",
        f"| Depth | {depth_ppm:.0f} ppm ({depth_pct:.3f}%) |",
        "",
    ]
    if notes:
        md_lines += ["## Notes", "", notes, ""]
    if contact_email:
        md_lines += [f"**Contact:** {contact_email}", ""]

    markdown = "\n".join(md_lines)

    payload = asdict(req)
    payload["ra_hms"] = ra_hms
    payload["dec_dms"] = dec_dms
    payload["depth_pct"] = round(depth_pct, 4)
    json_payload = json.dumps(payload, indent=2)

    return ObsRequestResult(request=req, markdown=markdown, json_payload=json_payload)


def format_obs_request(result: ObsRequestResult) -> str:
    """Return Markdown observation request string."""
    return result.markdown


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="obs_request_formatter",
        description="Format a ground-based follow-up observation request.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("tmag", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("--priority", default="MEDIUM")
    parser.add_argument("--notes", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = build_obs_request(
        args.tic_id, args.ra_deg, args.dec_deg, args.tmag,
        args.period_days, args.epoch_bjd, args.duration_hours, args.depth_ppm,
        priority=args.priority, notes=args.notes, contact_email=args.email,
    )
    if args.json:
        print(result.json_payload)
    else:
        print(format_obs_request(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
