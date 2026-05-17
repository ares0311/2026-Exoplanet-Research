"""Export candidates as a self-contained HTML gallery.

Produces a browsable HTML file with a sortable summary table and (optionally)
embedded base64 phase-fold plots per candidate when matplotlib is available.

Public API
----------
to_html_gallery(rows, *, title, plot_dir) -> str
write_html_gallery(rows, output_path, *, title, plot_dir) -> Path
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

_HTML_HEADER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #f8f9fa; }}
h1 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.8rem; text-align: left; }}
th {{ background: #343a40; color: white; cursor: pointer; }}
th:hover {{ background: #495057; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.grade-A {{ color: #27ae60; font-weight: bold; }}
.grade-B {{ color: #f39c12; font-weight: bold; }}
.grade-C {{ color: #e74c3c; font-weight: bold; }}
.grade-D {{ color: #7f0000; font-weight: bold; }}
.candidate-card {{ background: white; border: 1px solid #dee2e6;
  border-radius: 4px; margin: 1rem 0; padding: 1rem; }}
img {{ max-width: 600px; width: 100%; }}
</style>
<script>
function sortTable(n) {{
  var t = document.getElementById("candidates");
  var rows = Array.from(t.rows).slice(1);
  var asc = t.getAttribute("data-sort") == n;
  rows.sort(function(a, b) {{
    var x = a.cells[n].innerText, y = b.cells[n].innerText;
    return asc ? x.localeCompare(y, undefined, {{numeric: true}})
               : y.localeCompare(x, undefined, {{numeric: true}});
  }});
  rows.forEach(function(r) {{ t.appendChild(r); }});
  t.setAttribute("data-sort", asc ? "" : n);
}}
</script>
</head>
<body>
<h1>{title}</h1>
<p>{n_candidates} candidate(s)</p>
"""

_HTML_FOOTER = """\
</body>
</html>
"""


def _fpp(row: dict[str, Any]) -> float | None:
    v = row.get("scores", {}).get("false_positive_probability")
    if v is None:
        v = row.get("best_fpp")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _embed_plot(plot_path: Path) -> str:
    """Return an <img> tag with base64-encoded PNG, or empty string."""
    if not plot_path.exists():
        return ""
    data = base64.b64encode(plot_path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="phase fold">'


def to_html_gallery(
    rows: list[dict[str, Any]],
    *,
    title: str = "Candidate Gallery",
    plot_dir: Path | str | None = None,
) -> str:
    """Build a self-contained HTML gallery for a list of candidates.

    Args:
        rows: Candidate dicts from ``run_pipeline()`` output.
        title: Page heading.
        plot_dir: Directory containing PNG plots named
            ``<candidate_id>.png``.  If a matching file exists, it is
            embedded as a base64 image.

    Returns:
        HTML string.
    """
    if not rows:
        return "<p><em>No candidates.</em></p>\n"

    sorted_rows = sorted(rows, key=lambda r: _fpp(r) or 1.0)
    n = len(sorted_rows)

    parts = [_HTML_HEADER.format(title=html.escape(title), n_candidates=n)]

    # Summary table
    parts.append(
        '<h2>Summary</h2>'
        '<table id="candidates">'
        '<tr>'
        '<th onclick="sortTable(0)">Candidate</th>'
        '<th onclick="sortTable(1)">Target</th>'
        '<th onclick="sortTable(2)">Period (d)</th>'
        '<th onclick="sortTable(3)">FPP</th>'
        '<th onclick="sortTable(4)">Pathway</th>'
        '<th onclick="sortTable(5)">Rank</th>'
        '</tr>'
    )
    for row in sorted_rows:
        cid     = html.escape(str(row.get("candidate_id", "—")))
        target  = html.escape(str(row.get("target_id", "—")))
        period  = row.get("period_days")
        fpp_val = _fpp(row)
        pathway = html.escape(str(row.get("pathway") or row.get("best_pathway") or "—"))
        rank    = row.get("rank_score")
        pstr    = f"{period:.4f}" if isinstance(period, float) else "—"
        fstr    = f"{fpp_val:.4f}" if isinstance(fpp_val, float) else "—"
        rstr    = f"{rank:.3f}"   if isinstance(rank, float)   else "—"
        parts.append(
            f"<tr><td><a href='#{cid}'>{cid}</a></td><td>{target}</td>"
            f"<td>{pstr}</td><td>{fstr}</td><td>{pathway}</td><td>{rstr}</td></tr>"
        )
    parts.append("</table>")

    # Per-candidate cards
    parts.append("<h2>Candidates</h2>")
    for row in sorted_rows:
        cid     = str(row.get("candidate_id", "unknown"))
        target  = str(row.get("target_id", "—"))
        period  = row.get("period_days")
        fpp_val = _fpp(row)
        pathway = str(row.get("pathway") or row.get("best_pathway") or "—")

        plot_html = ""
        if plot_dir is not None:
            candidate_png = Path(plot_dir) / f"{cid}.png"
            plot_html = _embed_plot(candidate_png)

        parts.append(
            f'<div class="candidate-card" id="{html.escape(cid)}">'
            f"<h3>{html.escape(cid)}</h3>"
            f"<p><b>Target:</b> {html.escape(target)}</p>"
        )
        if isinstance(period, float):
            parts.append(f"<p><b>Period:</b> {period:.4f} d</p>")
        if isinstance(fpp_val, float):
            parts.append(f"<p><b>FPP:</b> {fpp_val:.4f}</p>")
        parts.append(f"<p><b>Pathway:</b> {html.escape(pathway)}</p>")
        if plot_html:
            parts.append(plot_html)
        parts.append("</div>")

    parts.append(_HTML_FOOTER)
    return "\n".join(parts)


def write_html_gallery(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Candidate Gallery",
    plot_dir: Path | str | None = None,
) -> Path:
    """Write the HTML gallery to a file.

    Args:
        rows: Candidate dicts.
        output_path: Destination HTML file.
        title: Page heading.
        plot_dir: Optional directory with PNG plots.

    Returns:
        Path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_html_gallery(rows, title=title, plot_dir=plot_dir))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_html_export",
        description="Export candidates as a self-contained HTML gallery.",
    )
    parser.add_argument("input", type=Path, metavar="FILE",
                        help="Pipeline JSON output file.")
    parser.add_argument("--output", type=Path, required=True, metavar="FILE",
                        help="Destination HTML file.")
    parser.add_argument("--plot-dir", type=Path, default=None, metavar="DIR",
                        help="Directory with PNG plots named <candidate_id>.png.")
    parser.add_argument("--title", default="Candidate Gallery", metavar="TITLE")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text())
    rows = data if isinstance(data, list) else [data]
    write_html_gallery(rows, args.output, title=args.title, plot_dir=args.plot_dir)
    print(f"Gallery written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
