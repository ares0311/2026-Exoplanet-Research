# PROJECT STATUS

## Status: Active Development
## Phase: Phase 6 — TTV Score, TOI Checker, Export, Alert Filter, Skills Guide Complete
## Last Updated: 2026-05-16

---

## What Is Complete

| Area | Key Files | Tests |
|------|-----------|-------|
| Scoring engine | `schemas`, `features`, `hypotheses`, `scoring`, `pathway` | 223 |
| Data pipeline | `fetch`, `clean`, `search`, `vet`, `calibration` | 254 |
| Provenance score | `fetch.py` — `compute_provenance_score()`; wired into `run_pipeline()` | 19 |
| CLI — transit scan | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output` | 24 |
| CLI — background automation | `cli.py` — `exo background-run-once`, `run-summary`, `sqlite-integrity`, etc. | 16 |
| Background automation module | `background/` — SQLite store, runner, priority, reports, fixtures, followup | (above) |
| ML Tier 1 | `ml/xgboost_scorer.py` | 45 |
| ML Tier 3 | `ml/stacking_scorer.py` | 22 |
| Kepler training | `fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py` | 34+25 |
| TESS training | `fetch_tess_toi.py`, `build_tess_training_data.py` | 11+38 |
| Combined training | `build_combined_training_data.py` | 13 |
| Evaluation | `evaluate_scorer.py` — ROC-AUC, F1, reliability diagram | 14 |
| Injection-recovery | `injection_recovery.py` | 25 |
| Star scanner | `star_scanner.py` — TIC priority ranking, background scan loop, JSON log | 38 |
| Candidate ranking | `rank_candidates.py` — composite rank score, Rich table, `--top N` | 12 |
| Batch scan | `batch_scan.py` — text/CSV input, incremental JSON output, `--resume` | 14 |
| Sector coverage | `sector_coverage.py` — query TESS sector availability without download | 10 |
| Depth chi-square | `features.py` — `depth_scatter_chi2_score`; wired into instrumental/planet hypotheses | 13 |
| Phase-fold plots | `plot_lc.py` — `phase_fold`, `plot_candidate`, `plot_all` | 11 |
| Watchlist | `watchlist.py` — persistent JSON watchlist for follow-up TIC IDs | 13 |
| Summary report | `summary_report.py` — Markdown report from batch_scan output | 14 |
| Transit timing variation | `features.py` — `transit_timing_variation_score`; wired into instrumental/planet hypotheses | 13 |
| TOI checker | `toi_checker.py` — ExoFOP TOI lookup; format_toi_result | 12 |
| Export candidates | `export_candidates.py` — `to_csv`, `to_markdown_table`, `to_summary_stats` | 13 |
| Alert filter | `alert_filter.py` — AND-logic threshold filtering over FPP/pathway/signals/SNR | 12 |
| CNN gate | `count_tess_labels.py` | — |
| Docs | `ML_SCORING.md`, `CNN_SPEC.md`, `DATA_SOURCES.md`, `DECISIONS.md`, `SCORING_MODEL.md §21-22`, `SKILLS_GUIDE.md` | — |
| Docs — automation | `BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`, `BACKGROUND_SEARCH_SQLITE_SCHEMA.md`, `SCHEDULER.md`, `SYSTEM_PROFILE.md` | — |
| README | 14-section rewrite with equations, MLA citations, submission guide, user guide | — |

**Total: 907 passing tests (+ 2 integration_live; 6 skipped without matplotlib)**

---

## Blocked

**ML Tier 2 — 1D CNN on phase-folded flux**
- Gate: 5,000+ labeled TESS light curves required
- Current CP count: check with `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`

---

## Next Actions

1. Run `python Skills/count_tess_labels.py` periodically to monitor CNN gate
2. Once gate opens: implement Tier 2 CNN per `docs/CNN_SPEC.md`
3. Use `toi_checker.py` before investing pipeline time on any new target
4. Track watchlist candidates through `batch_scan.py` + `alert_filter.py` → `export_candidates.py` for systematic follow-up
5. Consult `docs/SKILLS_GUIDE.md` for the full Skills discovery workflow

---

## Key Design Decisions in Effect

- Bayesian log-score model is default; XGBoost/ensemble are opt-in via `--scorer`
- `OptScore = float | None` — missing diagnostics are neutral (contribute 0)
- Conservative priors: 10% planet_candidate, 20% each false-positive class
- `None` feature scores fail threshold gates conservatively (§15 Guardrails)
- `provenance_score` computed from cadence, sector count, pipeline quality
- `depth_scatter_chi2_score` complements `depth_consistency_score` with error-weighted chi-square
- `transit_timing_variation_score` measures O-C RMS; high score → artifact evidence
- `toi_checker.py` should be consulted before investing pipeline time on any new target
- `alert_filter.py` applies AND-logic threshold filtering; FPP extracted from multiple dict shapes
- Never output "confirmed planet" — use "candidate signal"
- Background automation uses SQLite for durable state; JSON fixtures for offline testing
- Background automation obeys human-approval gate — no external submission without review
