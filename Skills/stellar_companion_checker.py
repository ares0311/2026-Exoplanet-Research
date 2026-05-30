from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CompanionCheckResult:
    detection_limit_arcsec: float
    companion_within_limit: bool
    separation_arcsec: float
    flag: str


def check_stellar_companion(
    separation_arcsec: float,
    tmag: float,
    contrast_mag: float,
    *,
    pixel_scale_arcsec: float = 21.0,
) -> CompanionCheckResult:
    """Check whether a stellar companion falls within the detection limit.

    detection_limit_arcsec = pixel_scale_arcsec * (1.0 + contrast_mag / 5.0)
    """
    if separation_arcsec < 0:
        return CompanionCheckResult(
            detection_limit_arcsec=0.0,
            companion_within_limit=False,
            separation_arcsec=separation_arcsec,
            flag="INVALID_SEPARATION",
        )

    if contrast_mag < 0:
        return CompanionCheckResult(
            detection_limit_arcsec=0.0,
            companion_within_limit=False,
            separation_arcsec=separation_arcsec,
            flag="INVALID_CONTRAST",
        )

    if tmag < 0:
        return CompanionCheckResult(
            detection_limit_arcsec=0.0,
            companion_within_limit=False,
            separation_arcsec=separation_arcsec,
            flag="INVALID_TMAG",
        )

    detection_limit_arcsec = pixel_scale_arcsec * (1.0 + contrast_mag / 5.0)
    companion_within_limit = separation_arcsec <= detection_limit_arcsec

    flag = "WITHIN_LIMIT" if companion_within_limit else "OUTSIDE_LIMIT"

    return CompanionCheckResult(
        detection_limit_arcsec=detection_limit_arcsec,
        companion_within_limit=companion_within_limit,
        separation_arcsec=separation_arcsec,
        flag=flag,
    )


def format_companion_result(result: CompanionCheckResult) -> str:
    """Return a Markdown table summarising the companion check result."""
    within_str = "Yes" if result.companion_within_limit else "No"
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Detection Limit (arcsec) | {result.detection_limit_arcsec:.3f} |",
        f"| Companion Within Limit | {within_str} |",
        f"| Separation (arcsec) | {result.separation_arcsec:.3f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a stellar companion falls within the TESS detection limit."
    )
    parser.add_argument(
        "separation_arcsec", type=float, help="Companion separation in arcseconds."
    )
    parser.add_argument("tmag", type=float, help="Target TESS magnitude.")
    parser.add_argument(
        "contrast_mag",
        type=float,
        help="Magnitude contrast between target and companion.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=21.0,
        help="Detector pixel scale in arcsec/pixel (default 21.0).",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    result = check_stellar_companion(
        args.separation_arcsec,
        args.tmag,
        args.contrast_mag,
        pixel_scale_arcsec=args.pixel_scale,
    )

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_companion_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
