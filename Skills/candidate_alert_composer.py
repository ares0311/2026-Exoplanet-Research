"""Compose structured plain-text alert messages for candidate signals."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class AlertMessage:
    tic_id: str
    subject: str
    body: str
    slack_text: str
    flag: str


_PATHWAY_LABELS: dict[str, str] = {
    "tfop_ready": "TFOP READY",
    "planet_hunters_discussion": "Planet Hunters Discussion",
    "kepler_archive_candidate": "Kepler Archive Candidate",
    "paper_or_preprint_candidate": "Paper/Preprint Candidate",
    "github_only_reproducibility": "GitHub Reproducibility Only",
    "known_object_annotation": "Known Object Annotation",
}


def compose_alert(
    candidate: dict,
    sender: str = "exo-toolkit",
) -> AlertMessage:
    """
    Compose a plain-text alert for a candidate signal.

    Expects keys: tic_id, period_days, depth_ppm, scores.false_positive_probability,
    scores.detection_confidence, pathway (optional).
    """
    tic_id = str(candidate.get("tic_id", "")).strip()
    if not tic_id:
        return AlertMessage(
            tic_id=tic_id, subject="", body="", slack_text="",
            flag="MISSING_TIC_ID",
        )

    period = candidate.get("period_days")
    depth = candidate.get("depth_ppm")
    scores = candidate.get("scores", {})
    fpp = scores.get("false_positive_probability", candidate.get("false_positive_probability"))
    dc = scores.get("detection_confidence", candidate.get("detection_confidence"))
    pathway = candidate.get("pathway", "unknown")
    pathway_label = _PATHWAY_LABELS.get(pathway, pathway)

    lines = [f"[{sender}] Transit Candidate Alert — TIC {tic_id}"]
    lines.append("=" * 60)
    if period is not None:
        lines.append(f"Period      : {period:.4f} d")
    if depth is not None:
        lines.append(f"Depth       : {depth:.0f} ppm")
    if fpp is not None:
        lines.append(f"FPP         : {fpp:.4f}")
    if dc is not None:
        lines.append(f"Detection   : {dc:.4f}")
    lines.append(f"Pathway     : {pathway_label}")
    lines.append("=" * 60)
    lines.append("Requires human review before any submission.")

    body = "\n".join(lines)
    if period:
        subject = f"Transit Candidate: TIC {tic_id} (P={period:.3f}d)"
    else:
        subject = f"Transit Candidate: TIC {tic_id}"

    # Slack-style compact message
    fpp_str = f"{fpp:.3f}" if fpp is not None else "N/A"
    period_str = f"{period:.3f}d" if period is not None else "N/A"
    slack_text = (
        f":telescope: *Transit Candidate* TIC {tic_id} | "
        f"P={period_str} | FPP={fpp_str} | {pathway_label}"
    )

    return AlertMessage(
        tic_id=tic_id,
        subject=subject,
        body=body,
        slack_text=slack_text,
        flag="OK",
    )


def format_alert_message(r: AlertMessage) -> str:
    return (
        f"**Subject:** {r.subject}\n\n"
        f"**Slack:** {r.slack_text}\n\n"
        f"**Body:**\n```\n{r.body}\n```\n"
        f"**Flag:** {r.flag}\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compose transit candidate alert message.")
    p.add_argument("candidate_json", help="JSON string or @file path with candidate dict")
    p.add_argument("--sender", default="exo-toolkit")
    args = p.parse_args()
    raw = args.candidate_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            candidate = json.load(f)
    else:
        candidate = json.loads(raw)
    r = compose_alert(candidate, sender=args.sender)
    print(format_alert_message(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
