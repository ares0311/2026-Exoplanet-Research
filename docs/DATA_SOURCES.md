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

Not yet integrated.  Future source for citizen-science transit candidates to
augment the TESS TOI training set.

---

## Caching Recommendations

- Light curves: cache to `~/.lightkurve-cache/` (Lightkurve default)
- KOI CSV: save to `data/kepler_koi.csv`; re-download quarterly
- TOI CSV: save to `data/tess_toi.csv`; re-download monthly (table grows)
- Trained models: save to `models/`; version by date or commit hash

---

## Authentication

For higher MAST rate limits, create a token at https://auth.mast.stsci.edu
and add to `~/.mast_api_token`.  Lightkurve reads it automatically.
