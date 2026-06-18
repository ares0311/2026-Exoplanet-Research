# Data Sources

This document lists the external data sources used by the toolkit, their
access methods, rate limits, and the scripts that fetch from them.

---

## MAST / Lightkurve — Light Curves

| Property | Value |
|----------|-------|
| Provider | Space Telescope Science Institute (STScI) |
| URL | https://mast.stsci.edu |
| Access | `lightkurve.search_lightcurve()` (wraps MAST API) |
| Auth | None required for public data |
| Rate limit | ~1,000 requests/day unauthenticated; use `~/.mast_api_token` for higher limits |
| Script | `fetch.py` (`fetch_lightcurve(tic_id, mission)`) |

**Preferred flux type**: `PDCSAP_FLUX` (systematics-corrected).  Fall back to
`SAP_FLUX` only when PDCSAP is unavailable.

**Missions supported**: `"TESS"`, `"Kepler"`, `"K2"`.

---

## NASA Exoplanet Archive — Kepler KOI Table

| Property | Value |
|----------|-------|
| Provider | NASA / IPAC |
| URL | https://exoplanetarchive.ipac.caltech.edu |
| Table | `cumulative` (DR25 Robovetter, Thompson et al. 2018) |
| Access | `astroquery.ipac.nexsci.nea_exoplanet.NasaExoplanetArchive` |
| Auth | None required |
| Rate limit | ~1,000 synchronous queries/day |
| Script | `Skills/fetch_kepler_tce.py` |

Columns fetched: `kepoi_name`, `koi_disposition`, `koi_model_snr`,
`koi_count`, `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`,
`koi_dikco_msky`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`.

Labels used: `CONFIRMED` → 1, `FALSE POSITIVE` → 0.
`CANDIDATE` excluded (noisy labels).

---

## ExoFOP-TESS — TESS TOI Disposition Table

| Property | Value |
|----------|-------|
| Provider | IPAC / Caltech |
| URL | https://exofop.ipac.caltech.edu/tess |
| Download | `download_toi.php?sort=toi&output=csv` |
| Auth | None required for the public TOI table |
| Rate limit | Not documented; cache locally after first download |
| Script | `Skills/fetch_tess_toi.py` |

Columns fetched: `TOI`, `TIC ID`, `TFOPWG Disposition`, `Period (days)`,
`Duration (hours)`, `Depth (mmag)`, `Planet Radius (R_Earth)`, `Planet SNR`,
`Number of Sectors`, `Stellar Radius (R_Sun)`, `Stellar Eff Temp (K)`,
`Stellar log(g) (cm/s^2)`, `TESS Mag`.

Labels used: `CP` → 1, `FP` / `EB` → 0.  `PC` excluded.

Gate check: `python Skills/count_tess_labels.py` monitors CP count for CNN Tier-2.

---

## CTOI / Community TOI

Not integrated into the default training pipeline. Treat CTOI/community
candidate tables as an opt-in source for citizen-science transit candidates
that can augment the TESS TOI training set only after an intentional fetch and
label assembly step.

The fixture-backed source contract is documented in
`docs/CTOI_SOURCE_CONTRACT.md`.

Source contract status before any default-training integration:

- fetch command lives under `Skills/` — implemented by
  `Skills/fetch_exofop_ctoi.py`
- default tests use committed fixtures or injected fetch functions, not live
  network calls
- the committed label-row fixture is
  `tests/fixtures/exofop_ctoi_labels_sample.json`
- labels map only externally reviewed dispositions into training labels
- uncertain/community-only candidate labels remain excluded from default
  supervised training
- opt-in label export uses
  `python Skills/fetch_exofop_ctoi.py --labels-output data/exofop_ctoi_labels.json`
- fetched tables are cached under `data/` and are not committed unless promoted
  as explicit fixtures
- docs record provider URL, auth requirements, rate-limit expectations, and
  column mappings before scorer training uses the source

---

## TESS TCE Source Probe

| Property | Value |
|----------|-------|
| Provider | STScI / MAST historical ExoMAST endpoint |
| Historical URL | `https://exo.mast.stsci.edu/api/v0.1/exoplanets/tce/` |
| Current status | Unavailable as of 2026-06-18: the endpoint returns HTTP 404 |
| Script | `Skills/tess_tce_fetcher.py` |

This source is **not approved** as a production T1-1 label source. The helper
now fails closed with `Flag: UNAVAILABLE` for the stale endpoint instead of
silently reporting an empty or invalid corpus. Do not plan a TESS TCE training
run from this source until a current provider contract, fields, label semantics,
rate limits, and offline fixtures are documented.

---

## Caching Recommendations

- Light curves: cache to `~/.lightkurve-cache/` (Lightkurve default)
- KOI CSV: save to `data/kepler_koi.csv`; re-download quarterly
- TOI CSV: save to `data/tess_toi.csv`; re-download monthly (table grows)
- Generated training corpora, including `data/tess_snippets.jsonl`, remain
  local runtime artifacts and are not committed.
- Generated split files and per-epoch training checkpoints remain local
  runtime artifacts under ignored `data/processed/` and `checkpoints/` paths.
- Validated production models, calibration metadata, registry entries, and
  reproducibility manifests: save to `models/`, version by date or commit
  hash, and commit after production-readiness validation.

---

## Authentication

For higher MAST rate limits, create a token at https://auth.mast.stsci.edu
and add to `~/.mast_api_token`.  Lightkurve reads it automatically.
