"""Build an interactive local browser UI for candidate review.

The UI is a self-contained HTML app. It can embed normalized candidate rows for
offline file viewing or fetch them from the local read-only candidate API. It
does not query live astronomy services and does not mutate source data.

Public API
----------
build_browser_ui(rows, *, title, api_base_url, embed_data, generated_at) -> str
write_browser_ui(rows, output_path, *, title, api_base_url, embed_data) -> Path
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Skills.candidate_api import candidate_to_payload, summary_payload
from Skills.candidate_dashboard_export import load_dashboard_rows, normalize_candidate


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _normalized_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        (normalize_candidate(row) for row in rows),
        key=lambda c: (
            c.false_positive_probability is None,
            (
                c.false_positive_probability
                if c.false_positive_probability is not None
                else 1.0
            ),
            c.candidate_id,
        ),
    )
    return {
        "summary": summary_payload(candidates),
        "candidates": [candidate_to_payload(candidate) for candidate in candidates],
    }


def _script() -> str:
    return """
const state = { candidates: [], selectedId: null };

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function unique(values) {
  return [...new Set(values.filter(Boolean))].sort();
}

function computeSummary(candidates) {
  const fpps = candidates
    .map(c => c.false_positive_probability)
    .filter(v => typeof v === "number");
  const risks = {};
  const pathways = {};
  candidates.forEach(c => {
    risks[c.risk_band || "unknown"] = (risks[c.risk_band || "unknown"] || 0) + 1;
    pathways[c.pathway || "unknown"] = (pathways[c.pathway || "unknown"] || 0) + 1;
  });
  const sorted = [...fpps].sort((a, b) => a - b);
  return {
    n_candidates: candidates.length,
    min_false_positive_probability: sorted.length ? sorted[0] : null,
    median_false_positive_probability: sorted.length ? sorted[Math.floor(sorted.length / 2)] : null,
    blocked_count: risks.blocked || 0,
    risk_counts: risks,
    pathway_counts: pathways
  };
}

function renderSummary(candidates) {
  const summary = computeSummary(candidates);
  document.getElementById("metric-count").textContent = summary.n_candidates;
  document.getElementById("metric-min-fpp").textContent =
    fmt(summary.min_false_positive_probability, 4);
  document.getElementById("metric-median-fpp").textContent =
    fmt(summary.median_false_positive_probability, 4);
  document.getElementById("metric-blocked").textContent = summary.blocked_count;
}

function populateFilters(candidates) {
  const risk = document.getElementById("risk-filter");
  const pathway = document.getElementById("pathway-filter");
  unique(candidates.map(c => c.risk_band)).forEach(value => {
    risk.insertAdjacentHTML(
      "beforeend",
      `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`
    );
  });
  unique(candidates.map(c => c.pathway)).forEach(value => {
    pathway.insertAdjacentHTML(
      "beforeend",
      `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`
    );
  });
}

function filteredCandidates() {
  const q = document.getElementById("search").value.toLowerCase();
  const risk = document.getElementById("risk-filter").value;
  const pathway = document.getElementById("pathway-filter").value;
  return state.candidates.filter(c => {
    const text = `${c.candidate_id} ${c.target_id} ${c.pathway} ${c.risk_band}`.toLowerCase();
    return (!q || text.includes(q)) &&
      (!risk || c.risk_band === risk) &&
      (!pathway || c.pathway === pathway);
  });
}

function renderTable() {
  const tbody = document.getElementById("candidate-rows");
  const candidates = filteredCandidates();
  tbody.innerHTML = candidates.map(c => `
    <tr data-id="${escapeHtml(c.candidate_id)}">
      <td>${escapeHtml(c.candidate_id)}</td>
      <td>${escapeHtml(c.target_id)}</td>
      <td>${fmt(c.period_days, 5)}</td>
      <td>${fmt(c.false_positive_probability, 4)}</td>
      <td><span class="tag ${escapeHtml(c.risk_band)}">${escapeHtml(c.risk_band)}</span></td>
      <td>${escapeHtml(c.pathway)}</td>
    </tr>
  `).join("");
  document.getElementById("filtered-count").textContent = candidates.length;
  [...tbody.querySelectorAll("tr")].forEach(row => {
    row.addEventListener("click", () => selectCandidate(row.dataset.id));
  });
  if (candidates.length && !candidates.some(c => c.candidate_id === state.selectedId)) {
    selectCandidate(candidates[0].candidate_id);
  }
}

function listItems(items, fallback) {
  const values = Array.isArray(items) && items.length ? items : [fallback];
  return values.map(item => `<li>${escapeHtml(item)}</li>`).join("");
}

function selectCandidate(candidateId) {
  state.selectedId = candidateId;
  const c = state.candidates.find(item => item.candidate_id === candidateId);
  const panel = document.getElementById("detail");
  if (!c) {
    panel.innerHTML = "<p>No candidate selected.</p>";
    return;
  }
  panel.innerHTML = `
    <h2>
      ${escapeHtml(c.candidate_id)}
      <span class="tag ${escapeHtml(c.risk_band)}">${escapeHtml(c.risk_band)}</span>
    </h2>
    <p>
      <b>Target:</b> ${escapeHtml(c.target_id)} |
      <b>Status:</b> ${escapeHtml(c.status)} |
      <b>Pathway:</b> ${escapeHtml(c.pathway)}
    </p>
    <p>
      <b>Period:</b> ${fmt(c.period_days, 5)} d |
      <b>Depth:</b> ${fmt(c.depth_ppm, 1)} ppm |
      <b>SNR:</b> ${fmt(c.snr, 2)}
    </p>
    <h3>Positive Evidence</h3>
    <ul>${listItems(c.positive_evidence, "No positive evidence supplied.")}</ul>
    <h3>False-Positive And Negative Evidence</h3>
    <ul>${listItems(c.negative_evidence, "No false-positive evidence supplied.")}</ul>
    <h3>Blocking Issues</h3>
    <ul>${listItems(c.blocking_issues, "No blocking issues supplied.")}</ul>
    <p class="source"><b>Source:</b> ${escapeHtml(c.source_file || "n/a")}</p>
  `;
  [...document.querySelectorAll("#candidate-rows tr")].forEach(row => {
    row.classList.toggle("selected", row.dataset.id === candidateId);
  });
}

async function loadData() {
  const embedded = document.getElementById("embedded-candidate-data");
  const apiBase = document.body.dataset.apiBase;
  if (embedded && embedded.textContent.trim()) {
    return JSON.parse(embedded.textContent);
  }
  if (apiBase) {
    const candidates = await fetch(`${apiBase.replace(/\\/$/, "")}/candidates`).then(r => r.json());
    return { candidates };
  }
  return { candidates: [] };
}

async function init() {
  try {
    const data = await loadData();
    state.candidates = data.candidates || [];
    renderSummary(state.candidates);
    populateFilters(state.candidates);
    ["search", "risk-filter", "pathway-filter"].forEach(id => {
      document.getElementById(id).addEventListener("input", renderTable);
    });
    renderTable();
  } catch (error) {
    document.getElementById("detail").innerHTML =
      `<p class="error">Could not load local candidate data: ${escapeHtml(error.message)}</p>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
"""


def _style() -> str:
    return """
:root {
  color-scheme: light;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body {
  margin: 0;
  color: #1f2933;
  background: #f5f7fa;
}
header {
  background: #ffffff;
  border-bottom: 1px solid #d9e2ec;
  padding: 20px 28px;
}
main {
  max-width: 1340px;
  margin: 0 auto;
  padding: 22px 28px 36px;
}
h1 {
  font-size: 28px;
  margin: 0 0 8px;
}
.meta,
.notice,
.source {
  color: #52606d;
}
.notice {
  margin: 12px 0 0;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}
.metric,
.panel,
.detail {
  background: #ffffff;
  border: 1px solid #d9e2ec;
  border-radius: 6px;
}
.metric {
  padding: 12px 14px;
}
.metric b {
  display: block;
  font-size: 24px;
  margin-top: 4px;
}
.controls {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) 180px 220px;
  gap: 10px;
  margin: 18px 0 12px;
}
input,
select {
  border: 1px solid #bcccdc;
  border-radius: 5px;
  font: inherit;
  padding: 9px 10px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(520px, 1.25fr) minmax(340px, 0.75fr);
  gap: 16px;
}
.panel {
  overflow: auto;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th,
td {
  border-bottom: 1px solid #e4e7eb;
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #e6eef8;
}
tr {
  cursor: pointer;
}
tr:hover,
tr.selected {
  background: #f0f4f8;
}
.detail {
  padding: 16px;
}
.tag {
  background: #f0f4f8;
  border: 1px solid #bcccdc;
  border-radius: 999px;
  display: inline-block;
  font-size: 12px;
  padding: 2px 8px;
}
.blocked {
  border-color: #b42318;
  color: #b42318;
}
.high-fpp,
.elevated-fpp {
  border-color: #b54708;
  color: #b54708;
}
.low-fpp {
  border-color: #027a48;
  color: #027a48;
}
.error {
  color: #b42318;
}
@media (max-width: 900px) {
  .layout,
  .controls {
    grid-template-columns: 1fr;
  }
  main,
  header {
    padding-left: 16px;
    padding-right: 16px;
  }
}
"""


def build_browser_ui(
    rows: list[dict[str, Any]],
    *,
    title: str = "Candidate Browser",
    api_base_url: str | None = None,
    embed_data: bool = True,
    generated_at: str | None = None,
) -> str:
    """Build a self-contained browser UI for candidate review."""
    generated_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    payload = _normalized_payload(rows) if embed_data else {"summary": {}, "candidates": []}
    embedded_json = json.dumps(payload, sort_keys=True).replace("<", "\\u003c")
    api_attr = f' data-api-base="{_escape(api_base_url)}"' if api_base_url else ""
    embedded_block = (
        '<script type="application/json" id="embedded-candidate-data">'
        f"{embedded_json}"
        "</script>"
        if embed_data
        else ""
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(title)}</title>",
            "<style>",
            _style(),
            "</style>",
            "</head>",
            f"<body{api_attr}>",
            "<header>",
            f"<h1>{_escape(title)}</h1>",
            f'<p class="meta">Generated { _escape(generated_at) }.</p>',
            '<p class="notice">Local review UI for candidate signals and '
            "follow-up targets. No discovery or external validation claim is made.</p>",
            "</header>",
            "<main>",
            '<section class="metrics" aria-label="summary metrics">',
            '<div class="metric">Candidates<b id="metric-count">0</b></div>',
            '<div class="metric">Minimum FPP<b id="metric-min-fpp">n/a</b></div>',
            '<div class="metric">Median FPP<b id="metric-median-fpp">n/a</b></div>',
            '<div class="metric">Blocked<b id="metric-blocked">0</b></div>',
            "</section>",
            '<section class="controls" aria-label="candidate filters">',
            '<input id="search" type="search" placeholder="Search candidates, targets, pathways">',
            '<select id="risk-filter"><option value="">All risks</option></select>',
            '<select id="pathway-filter"><option value="">All pathways</option></select>',
            "</section>",
            '<p class="meta"><span id="filtered-count">0</span> shown after filters.</p>',
            '<section class="layout">',
            '<div class="panel">',
            "<table>",
            "<thead><tr>",
            "<th>Candidate</th><th>Target</th><th>Period (d)</th>",
            "<th>FPP</th><th>Risk</th><th>Pathway</th>",
            "</tr></thead>",
            '<tbody id="candidate-rows"></tbody>',
            "</table>",
            "</div>",
            '<aside class="detail" id="detail"><p>No candidate selected.</p></aside>',
            "</section>",
            "</main>",
            embedded_block,
            "<script>",
            _script(),
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def write_browser_ui(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Candidate Browser",
    api_base_url: str | None = None,
    embed_data: bool = True,
    generated_at: str | None = None,
) -> Path:
    """Write the browser UI HTML and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_browser_ui(
            rows,
            title=title,
            api_base_url=api_base_url,
            embed_data=embed_data,
            generated_at=generated_at,
        )
    )
    return path


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_browser_ui",
        description="Build an interactive local candidate browser HTML file.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--output", type=Path, required=True, metavar="HTML")
    parser.add_argument("--title", default="Candidate Browser")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-mode", action="store_true")
    args = parser.parse_args(argv)

    rows = load_dashboard_rows(args.files)
    write_browser_ui(
        rows,
        args.output,
        title=args.title,
        api_base_url=args.api_base_url,
        embed_data=not args.api_mode,
    )
    print(f"Browser UI written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
