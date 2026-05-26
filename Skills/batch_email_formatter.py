"""Format alert emails for a batch of exoplanet candidates.

Produces both plain-text and HTML email bodies from a list of candidate
result dicts (e.g. from batch_scan / star_scanner output).  Does NOT send
any email — that is left to the caller's SMTP layer.

Public API
----------
EmailMessage(subject, plain_text, html, n_candidates, flag)
format_batch_email(candidates, *, sender, recipients, title,
                   fpp_threshold, include_html) -> EmailMessage
format_single_candidate_email(row, *, sender, recipients) -> EmailMessage
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class EmailMessage:
    subject: str
    plain_text: str
    html: str
    n_candidates: int
    flag: str  # "OK" | "NO_CANDIDATES" | "EMPTY"


def _fpp(row: dict) -> float | None:
    for k in ("best_fpp", "false_positive_probability"):
        if k in row:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    scores = row.get("scores", {})
    if isinstance(scores, dict):
        v = scores.get("false_positive_probability")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _fmt_date() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><style>
body {{ font-family: Arial, sans-serif; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
tr:nth-child(even) {{ background-color: #fafafa; }}
.good {{ color: #2e7d32; font-weight: bold; }}
.warn {{ color: #e65100; font-weight: bold; }}
</style></head>
<body>
<h2>{title}</h2>
<p>Generated: {date}</p>
<p>Found <strong>{n}</strong> candidate(s) with FPP &lt; {threshold:.2f}.</p>
{table}
<hr>
<p style="font-size:11px;color:#888;">Sent by exo-toolkit. Do not reply to this address.</p>
</body>
</html>
"""

_TABLE_HEADER = (
    "<table>\n"
    "<tr><th>TIC ID</th><th>Period (d)</th><th>FPP</th>"
    "<th>Pathway</th><th>Signals</th></tr>\n"
)


def _candidate_table_html(rows: list[dict]) -> str:
    lines = [_TABLE_HEADER]
    for row in rows:
        tic = row.get("tic_id", "—")
        period = row.get("best_period_days", "—")
        fpp_val = _fpp(row)
        fpp_str = f"{fpp_val:.3f}" if fpp_val is not None else "—"
        pathway = row.get("best_pathway", row.get("pathway", "—"))
        n_sig = row.get("n_signals", "—")
        css = "good" if (fpp_val is not None and fpp_val < 0.10) else "warn"
        lines.append(
            f"<tr><td>{tic}</td>"
            f"<td>{period}</td>"
            f"<td class='{css}'>{fpp_str}</td>"
            f"<td>{pathway}</td>"
            f"<td>{n_sig}</td></tr>\n"
        )
    lines.append("</table>")
    return "".join(lines)


def _candidate_table_plain(rows: list[dict]) -> str:
    header = f"{'TIC ID':<15} {'Period (d)':<12} {'FPP':<8} {'Pathway':<30} {'Signals'}"
    sep = "-" * 80
    lines = [header, sep]
    for row in rows:
        tic = str(row.get("tic_id", "—"))
        period = str(row.get("best_period_days", "—"))
        fpp_val = _fpp(row)
        fpp_str = f"{fpp_val:.3f}" if fpp_val is not None else "—"
        pathway = str(row.get("best_pathway", row.get("pathway", "—")))
        n_sig = str(row.get("n_signals", "—"))
        lines.append(f"{tic:<15} {period:<12} {fpp_str:<8} {pathway:<30} {n_sig}")
    return "\n".join(lines)


def format_batch_email(
    candidates: list[dict],
    *,
    sender: str = "exo-toolkit@noreply.example",
    recipients: list[str] | None = None,
    title: str = "Exoplanet Candidate Alert",
    fpp_threshold: float = 0.30,
    include_html: bool = True,
) -> EmailMessage:
    """Format a batch alert email for multiple candidates.

    Args:
        candidates: List of candidate result dicts.
        sender: From address (for header reference only — no email sent).
        recipients: List of recipient addresses (for header reference only).
        title: Email subject / heading.
        fpp_threshold: Only include candidates with FPP below this value.
        include_html: If False, HTML body is left empty.

    Returns:
        :class:`EmailMessage`.
    """
    if not candidates:
        return EmailMessage("", "", "", 0, "EMPTY")

    filtered = [r for r in candidates if _fpp(r) is None or (_fpp(r) or 1.0) < fpp_threshold]
    n = len(filtered)

    if n == 0:
        plain = f"No candidates passed FPP < {fpp_threshold:.2f} threshold."
        return EmailMessage(title, plain, "", 0, "NO_CANDIDATES")

    date = _fmt_date()
    subject = f"{title} — {n} candidate(s) found ({date})"

    plain_lines = [
        subject,
        "=" * len(subject),
        "",
        f"Generated: {date}",
        f"Candidates with FPP < {fpp_threshold:.2f}: {n}",
        "",
        _candidate_table_plain(filtered),
        "",
        "---",
        "Sent by exo-toolkit.",
    ]
    plain_text = "\n".join(plain_lines)

    html = ""
    if include_html:
        table = _candidate_table_html(filtered)
        html = _HTML_TEMPLATE.format(
            title=title, date=date, n=n, threshold=fpp_threshold, table=table
        )

    return EmailMessage(
        subject=subject,
        plain_text=plain_text,
        html=html,
        n_candidates=n,
        flag="OK",
    )


def format_single_candidate_email(
    row: dict,
    *,
    sender: str = "exo-toolkit@noreply.example",
    recipients: list[str] | None = None,
) -> EmailMessage:
    """Format a single-candidate alert email.

    Args:
        row: Single candidate result dict.
        sender: From address (for header reference only).
        recipients: Recipient addresses (for header reference only).

    Returns:
        :class:`EmailMessage`.
    """
    tic = row.get("tic_id", "unknown")
    period = row.get("best_period_days", "—")
    fpp_val = _fpp(row)
    fpp_str = f"{fpp_val:.4f}" if fpp_val is not None else "unknown"
    pathway = row.get("best_pathway", row.get("pathway", "—"))
    date = _fmt_date()
    subject = f"Candidate Alert: TIC {tic} — FPP {fpp_str} ({date})"

    plain_lines = [
        subject,
        "=" * len(subject),
        "",
        f"TIC ID        : {tic}",
        f"Period (days) : {period}",
        f"FPP           : {fpp_str}",
        f"Pathway       : {pathway}",
        f"N Signals     : {row.get('n_signals', '—')}",
        "",
        "---",
        "Sent by exo-toolkit.",
    ]
    plain_text = "\n".join(plain_lines)

    html = (
        f"<!DOCTYPE html><html><body>"
        f"<h2>Candidate Alert: TIC {tic}</h2>"
        f"<p><b>Period:</b> {period} days<br>"
        f"<b>FPP:</b> {fpp_str}<br>"
        f"<b>Pathway:</b> {pathway}</p>"
        f"<hr><p style='font-size:11px;color:#888;'>Sent by exo-toolkit.</p>"
        f"</body></html>"
    )

    return EmailMessage(
        subject=subject,
        plain_text=plain_text,
        html=html,
        n_candidates=1,
        flag="OK",
    )


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="batch_email_formatter",
        description="Format batch alert emails for exoplanet candidates.",
    )
    parser.add_argument("results_json", type=str)
    parser.add_argument("--fpp-threshold", type=float, default=0.30)
    parser.add_argument("--title", type=str, default="Exoplanet Candidate Alert")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args(argv)

    with open(args.results_json) as f:
        data = json.load(f)

    candidates = data if isinstance(data, list) else [data]
    msg = format_batch_email(
        candidates,
        title=args.title,
        fpp_threshold=args.fpp_threshold,
        include_html=not args.no_html,
    )
    print(f"Subject: {msg.subject}")
    print()
    print(msg.plain_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
