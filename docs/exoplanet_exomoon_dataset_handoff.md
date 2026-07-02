# Exoplanet and Exomoon Dataset Handoff for Coding Agent

## Goal

Build a practical data pipeline for training AI models to identify transiting exoplanet candidates and rank possible exomoon-like anomalies.

Core distinction:

- Exoplanets: large public labeled datasets exist.
- Exomoons: no large confirmed-positive real labeled dataset exists. Treat exomoon work as residual/anomaly ranking only, not ordinary supervised classification.

Primary deliverable:

- Build a working exoplanet classifier/ranker first.
- Keep exomoon logic as a separate Track B module.
- Do not use synthetic data.
- Do not download or stage more than about 100 GB of training data at one time without explicit user approval.
- Delete raw training light-curve files after processed training artifacts and trained model outputs are created and verified.

## API Keys

No API keys are needed for the recommended public datasets.

| Source | API key needed? | Use |
|---|---:|---|
| NASA Exoplanet Archive TAP | No | Labels, KOI table, TOI table, confirmed planets |
| MAST public Kepler/TESS data | No | Light curves, target pixel files, data validation files |
| ExoFOP-TESS public TOI CSV | No | TOI catalog and follow-up status |
| MAST exclusive/proprietary data | Yes | Not needed for this project |

## Authoritative Source Links

NASA Exoplanet Archive:

- Main archive: https://exoplanetarchive.ipac.caltech.edu/
- TAP guide: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
- Programmatic interfaces: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
- Kepler KOI columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
- TESS TOI columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html
- Planetary Systems Composite Parameters columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
- TESS Project Candidates documentation: https://exoplanetarchive.ipac.caltech.edu/docs/TESSMission.html

MAST / STScI:

- MAST Kepler archive: https://archive.stsci.edu/missions-and-data/kepler
- MAST Kepler public light curves: https://archive.stsci.edu/kepler/publiclightcurves.html
- MAST TESS archive: https://archive.stsci.edu/missions-and-data/tess
- TESS bulk downloads: https://archive.stsci.edu/tess/bulk_downloads.html
- TESS FFI/TP/LC/DV bulk scripts: https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html
- Astroquery MAST observations docs: https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html
- Lightkurve docs: https://lightkurve.github.io/lightkurve/

ExoFOP-TESS:

- ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
- TOI releases: https://tess.mit.edu/toi-releases/
- TOI release notes / CSV guidance: https://tess.mit.edu/toi-releases/toi-release-notes/
- Public TOI CSV endpoint: https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv

ML references:

- NASA ExoMiner: https://github.com/nasa/Exominer
- Google AstroNet / exoplanet-ml: https://github.com/google-research/exoplanet-ml/tree/master/exoplanet-ml/astronet
- AstroNet paper: https://arxiv.org/abs/1712.05044
- AstroNet-Triage-v2 for TESS: https://arxiv.org/abs/2301.01371

Exomoon references:

- Hunt for Exomoons with Kepler I: https://arxiv.org/abs/1201.0752
- HEK II: https://arxiv.org/abs/1301.1853
- HEK V survey: https://arxiv.org/abs/1503.05555
- Exomoon CNN candidate paper: https://academic.oup.com/mnras/article/508/2/2620/6373454

## Resource Access Contract

No guessing rule:

- Do not assume a table, column, label value, product type, or URL works because it appears in this document.
- Verify each resource programmatically before using it.
- If a required resource cannot be accessed, a required column is missing, or a schema has changed, stop and report the exact failure.
- Do not silently substitute Kaggle mirrors, blog datasets, old notebooks, scraped HTML tables, or synthetic data.

Required access methods:

| Resource | Exact access method | Verification before use | If verification fails |
|---|---|---|---|
| NASA Exoplanet Archive KOI labels | TAP query against table `cumulative` | Query `TAP_SCHEMA.columns` and confirm required columns | Stop; do not infer renamed columns |
| NASA Exoplanet Archive TESS TOIs | TAP query against table `toi` | Query `TAP_SCHEMA.columns` and confirm required columns | Stop; do not infer renamed columns |
| NASA confirmed planets | TAP query against table `pscomppars` | Query `TAP_SCHEMA.columns` and confirm required columns | Stop; use only after schema is verified |
| ExoFOP-TESS public TOI CSV | `pandas.read_csv` from the public CSV endpoint | Confirm non-empty table and preserve actual headers | Stop; do not scrape the website |
| Kepler light curves | Lightkurve search/download using `KIC <kepid>` from verified KOI table | Confirm at least one returned product and inspect FITS columns | Skip target or stop if systematic |
| TESS light curves | Lightkurve search/download using `TIC <tid>` from verified TOI table | Confirm at least one returned product and inspect FITS columns | Skip target or stop if systematic |
| NASA ExoMiner | GitHub repository only | Verify repo, license, and preprocessing assumptions | Use as reference only if weights/contract are unclear |
| Google AstroNet | GitHub repository only | Verify repo, license, and preprocessing assumptions | Use as reference only if weights/contract are unclear |

Minimum access smoke test:

```python
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from lightkurve import search_lightcurve

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOFOP_TOI_CSV = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

REQUIRED_KOI_COLUMNS = {
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
}

REQUIRED_TOI_COLUMNS = {
    "tid",
    "toi",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
}

def tap_csv(query: str) -> pd.DataFrame:
    response = requests.get(
        TAP_SYNC,
        params={"query": query, "format": "csv"},
        timeout=120,
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def tap_columns(table_name: str) -> set[str]:
    query = (
        "select column_name from TAP_SCHEMA.columns "
        f"where table_name = '{table_name}'"
    )
    df = tap_csv(query)
    return set(df["column_name"].astype(str))

def require_columns(table_name: str, required: set[str]) -> None:
    available = tap_columns(table_name)
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            f"Table {table_name} is missing required columns: {missing}. "
            "Stop; do not guess renamed columns."
        )

def smoke_test_catalogs() -> tuple[int, int]:
    require_columns("cumulative", REQUIRED_KOI_COLUMNS)
    require_columns("toi", REQUIRED_TOI_COLUMNS)

    koi = tap_csv(
        "select top 5 kepid,kepoi_name,koi_disposition,koi_period,"
        "koi_time0bk,koi_duration from cumulative"
    )
    toi = tap_csv(
        "select top 5 tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,"
        "pl_trandurh from toi"
    )
    if koi.empty or toi.empty:
        raise RuntimeError("KOI or TOI smoke-test query returned no rows.")

    exofop = pd.read_csv(EXOFOP_TOI_CSV, nrows=5)
    if exofop.empty:
        raise RuntimeError("ExoFOP public TOI CSV returned no rows.")

    return int(koi.iloc[0]["kepid"]), int(toi.iloc[0]["tid"])

def smoke_test_lightkurve(kepid: int, tid: int) -> None:
    kepler_search = search_lightcurve(f"KIC {kepid}", mission="Kepler")
    tess_search = search_lightcurve(f"TIC {tid}", mission="TESS")
    if len(kepler_search) == 0:
        raise RuntimeError(f"No Kepler light curves found for KIC {kepid}.")
    if len(tess_search) == 0:
        raise RuntimeError(f"No TESS light curves found for TIC {tid}.")

if __name__ == "__main__":
    kepid, tid = smoke_test_catalogs()
    smoke_test_lightkurve(kepid, tid)
    print("Resource access smoke test passed.")
```

Run this before implementing the full downloader. If it fails, fix the access issue or report the exact blocker.

## Recommended Data Strategy

Use two tracks.

Track A: working exoplanet classifier/ranker.

1. Train the first supervised model on Kepler DR25 KOI/TCE labels.
2. Download only the matching Kepler light curves needed for the current batch.
3. Validate on held-out Kepler systems by target, not by row, to avoid leakage.
4. Transfer/evaluate on a limited TESS TOI subset after the Kepler baseline works.
5. Keep the local working data footprint under about 100 GB unless the user approves more.
6. After training and verification, delete raw downloaded training light curves unless they are needed to reproduce a failed run.

Track B: exomoon residual/anomaly ranking.

1. Use only real light curves and real catalog metadata.
2. Run after a planet-only transit model has been fit.
3. Rank residual anomalies, TTV/TDV patterns, and repeatability for human review.
4. Do not train an exomoon supervised classifier because no large real confirmed-positive label set exists.

Do not market or represent the exomoon component as a confirmed-exomoon supervised classifier.

## Storage and Data Retention Rules

The user has about 400 GB free but wants this project to stay near a 100 GB working-data cap.

Hard requirements:

- Never bulk-download all Kepler/TESS archive products.
- Never use MAST sector-wide TESS bulk scripts unless the coding agent first estimates disk use and asks the user.
- Keep catalog tables, manifests, processed feature arrays, model weights, metrics, and logs.
- Delete raw downloaded FITS light-curve files after they have been converted into processed training artifacts and the run is verified.
- Keep enough metadata to re-download the exact raw files later: source URL, mission, target ID, product ID if available, file name, checksum if available, and download timestamp.
- Add a disk-usage guard before and after every download batch.

Example guard:

```python
from pathlib import Path
import shutil

MAX_WORKING_BYTES = 100 * 1024**3

def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

def assert_under_cap(path: str | Path, cap_bytes: int = MAX_WORKING_BYTES) -> None:
    used = directory_size_bytes(path)
    if used > cap_bytes:
        raise RuntimeError(
            f"Working data directory is {used / 1024**3:.1f} GB, "
            f"above cap of {cap_bytes / 1024**3:.1f} GB. Stop and ask user."
        )

def free_space_gb(path: str | Path = ".") -> float:
    return shutil.disk_usage(path).free / 1024**3
```

Suggested retention policy:

| Artifact | Keep after training? | Reason |
|---|---:|---|
| Raw catalog snapshots | Yes | Small and needed for provenance |
| Training manifest | Yes | Required for reproducibility |
| Processed folded/normalized arrays | Yes, if compact | Faster reuse than raw FITS |
| Raw FITS light curves | No, delete after verified processing | Large and re-downloadable |
| Model weights/checkpoints | Yes | Primary result |
| Metrics, plots, config, logs | Yes | Evaluation and audit trail |

## Pre-Trained Model Guidance

There are useful public model/code references, but do not assume there is a maintained plug-and-play pretrained model that can be dropped into this repo and trusted.

Recommended use:

- Use NASA ExoMiner and Google AstroNet as architecture/preprocessing references.
- Reproduce a small local baseline using the project's own downloaded Kepler labels and light curves.
- Only use external pretrained weights if the agent verifies the exact weights, license, training labels, preprocessing contract, input shape, and expected calibration.
- If those details are not verified, treat the external project as reference code, not as a production dependency.

Access checks for reference repositories:

```bash
git ls-remote https://github.com/nasa/Exominer.git HEAD
```

```bash
git ls-remote https://github.com/google-research/exoplanet-ml.git HEAD
```

If these commands fail, do not use the repositories in this phase. If they succeed, record the commit hash used for review. Do not import code or weights into the project until license and preprocessing assumptions are reviewed.

Avoid these rabbit holes:

- Do not spend time trying to make an unverified old notebook/model checkpoint the core pipeline.
- Do not train on Kaggle mirrors when the primary NASA archive can be queried directly.
- Do not mix Kepler and TESS blindly before a Kepler-only baseline works.
- Do not use synthetic injected moons or synthetic injected planets for this phase.

## NASA Exoplanet Archive TAP Basics

Base synchronous TAP endpoint:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync
```

Basic URL pattern:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL_QUERY>&format=csv
```

Use `requests` with query params instead of hand-building encoded URLs.

Command-line smoke tests:

```bash
curl --get 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync' \
  --data-urlencode "query=select top 5 kepid,kepoi_name,koi_disposition from cumulative" \
  --data-urlencode "format=csv"
```

```bash
curl --get 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync' \
  --data-urlencode "query=select top 5 tid,toi,tfopwg_disp from toi" \
  --data-urlencode "format=csv"
```

```bash
curl --get 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync' \
  --data-urlencode "query=select column_name,datatype from TAP_SCHEMA.columns where table_name = 'cumulative'" \
  --data-urlencode "format=csv"
```

```python
from io import StringIO

import pandas as pd
import requests

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap_csv(query: str) -> pd.DataFrame:
    response = requests.get(
        TAP_SYNC,
        params={"query": query, "format": "csv"},
        timeout=120,
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
```

Schema discovery:

```sql
select *
from TAP_SCHEMA.columns
where table_name = 'cumulative'
```

```sql
select *
from TAP_SCHEMA.columns
where table_name = 'toi'
```

Fail-closed requirements:

- Query `TAP_SCHEMA.columns` before every new source table is used.
- Compare required columns by exact name.
- If a required column is missing, stop and report the missing column list.
- Do not infer replacement columns from similar names without user approval.
- Query table row counts at download time and store them in `metadata/source_snapshots.json`.

Row-count queries:

```sql
select count(*) as n_rows from cumulative
```

```sql
select count(*) as n_rows from toi
```

```sql
select count(*) as n_rows from pscomppars
```

Required `metadata/source_snapshots.json` shape:

```json
{
  "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "sources": [
    {
      "name": "nasa_exoplanet_archive_cumulative",
      "access_method": "tap",
      "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
      "table": "cumulative",
      "query": "select ... from cumulative",
      "row_count_at_download": 0,
      "required_columns_verified": true,
      "required_columns": [],
      "actual_columns": []
    }
  ]
}
```

## Dataset 1: Kepler KOI Cumulative Table

Use this as the primary supervised label source.

Table name:

```text
cumulative
```

Download all columns:

```python
koi = tap_csv("select * from cumulative")
koi.to_parquet("data/raw/nasa_exoplanet_archive/koi_cumulative.parquet", index=False)
```

Access validation:

```python
required = {
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
}
available = tap_columns("cumulative")
missing = required - available
if missing:
    raise RuntimeError(f"Missing KOI columns: {sorted(missing)}")
```

Minimum production query:

```sql
select
  kepid,
  kepoi_name,
  kepler_name,
  koi_disposition,
  koi_pdisposition,
  koi_score,
  koi_period,
  koi_time0bk,
  koi_duration,
  koi_depth,
  koi_prad,
  koi_teq,
  koi_insol,
  koi_model_snr,
  koi_steff,
  koi_slogg,
  koi_srad,
  ra,
  dec,
  koi_kepmag
from cumulative
```

Expected schema:

| Column | Meaning | Use |
|---|---|---|
| `kepid` | Kepler Input Catalog ID | Join key to Kepler light curves |
| `kepoi_name` | Kepler Object of Interest name | Candidate/event ID |
| `kepler_name` | Confirmed planet name, if any | Confirmed-name reference |
| `koi_disposition` | `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE` | Primary label |
| `koi_pdisposition` | Pipeline disposition | Secondary label/check |
| `koi_score` | Disposition score | Confidence/weight |
| `koi_period` | Orbital period, days | Folding/model feature |
| `koi_time0bk` | Transit epoch, BKJD | Folding/model feature |
| `koi_duration` | Transit duration, hours | Folding/model feature |
| `koi_depth` | Transit depth, ppm | Feature |
| `koi_prad` | Planet radius, Earth radii | Feature/metadata |
| `koi_teq` | Equilibrium temperature | Feature/metadata |
| `koi_insol` | Insolation flux | Feature/metadata |
| `koi_model_snr` | Transit model signal-to-noise | Feature/quality |
| `koi_steff` | Stellar effective temperature | Host-star feature |
| `koi_slogg` | Stellar log g | Host-star feature |
| `koi_srad` | Stellar radius | Host-star feature |
| `ra` | Right ascension | Crossmatch/search |
| `dec` | Declination | Crossmatch/search |
| `koi_kepmag` | Kepler magnitude | Photometric quality feature |

Recommended initial label mapping:

```python
def map_koi_label(row):
    disp = row["koi_disposition"]
    score = row.get("koi_score")

    if disp == "CONFIRMED":
        return "planet_confirmed"
    if disp == "FALSE POSITIVE":
        return "false_positive"
    if disp == "CANDIDATE":
        if pd.notna(score) and score >= 0.9:
            return "planet_candidate_high_confidence"
        return "planet_candidate_weak"
    return "unknown"
```

For the first supervised model:

- Positive class: `CONFIRMED` plus high-confidence `CANDIDATE`.
- Negative class: `FALSE POSITIVE`.
- Holdout/weak class: low-confidence `CANDIDATE`.

## Dataset 2: Kepler Light Curves From MAST

Join key:

```text
KOI kepid -> MAST target name "KIC <kepid>"
```

Recommended Python path:

```python
from lightkurve import search_lightcurve

def download_kepler_lightcurves(kepid: int, download_dir: str):
    result = search_lightcurve(f"KIC {kepid}", mission="Kepler")
    if len(result) == 0:
        raise RuntimeError(f"No Kepler light curve search results for KIC {kepid}")
    return result.download_all(download_dir=download_dir)
```

Before downloading a batch:

```python
def kepler_search_metadata(kepid: int) -> list[dict]:
    result = search_lightcurve(f"KIC {kepid}", mission="Kepler")
    if len(result) == 0:
        return []
    return [dict(row) for row in result.table]
```

Write the search metadata to `metadata/download_manifest.jsonl` before downloading. This is required because raw FITS files should be deleted after verified processing, and the project must preserve enough information to re-find the products.

After downloading one file, open it with `astropy.io.fits` and verify that the light curve extension contains at least:

```text
TIME
QUALITY
```

Then use `PDCSAP_FLUX` if present. If `PDCSAP_FLUX` is absent for a product, either skip that product or explicitly route it through a raw-flux preprocessing path. Do not silently substitute `SAP_FLUX`.

Typical FITS light curve columns:

| Column | Meaning | Use |
|---|---|---|
| `TIME` | Time coordinate | Time series input |
| `CADENCENO` | Cadence number | Quality/debug |
| `SAP_FLUX` | Raw/simple aperture photometry flux | Raw input option |
| `SAP_FLUX_ERR` | SAP flux uncertainty | Weighting |
| `PDCSAP_FLUX` | Systematics-corrected flux | Recommended first ML input |
| `PDCSAP_FLUX_ERR` | Corrected flux uncertainty | Weighting |
| `QUALITY` | Bitmask quality flags | Mask bad cadences |
| `MOM_CENTR1`, `MOM_CENTR2` | Centroid estimates | False-positive diagnostics |
| `PSF_CENTR1`, `PSF_CENTR2` | PSF centroid estimates, if present | False-positive diagnostics |

Recommended preprocessing:

1. Use `PDCSAP_FLUX` initially.
2. Remove cadences where `QUALITY != 0` unless experimenting with robust masking.
3. Normalize each quarter separately.
4. Fold by `koi_period` and `koi_time0bk`.
5. Generate both global and local transit views, following AstroNet-style baselines.

## Dataset 3: TESS TOI Table

Use this after Kepler for transfer learning, modern validation, and TESS-specific triage.

NASA Exoplanet Archive table:

```text
toi
```

Download all columns:

```python
toi = tap_csv("select * from toi")
toi.to_parquet("data/raw/nasa_exoplanet_archive/tess_toi.parquet", index=False)
```

Access validation:

```python
required = {
    "tid",
    "toi",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
}
available = tap_columns("toi")
missing = required - available
if missing:
    raise RuntimeError(f"Missing TOI columns: {sorted(missing)}")
```

Minimum useful query:

```sql
select
  tid,
  toi,
  toipfx,
  tfopwg_disp,
  pl_orbper,
  pl_tranmid,
  pl_trandurh,
  pl_trandep,
  pl_rade,
  st_tmag,
  ra,
  dec
from toi
```

Expected schema:

| Column | Meaning | Use |
|---|---|---|
| `tid` | TESS Input Catalog ID | Join key to TESS light curves |
| `toi` | TESS Object of Interest ID | Candidate/event ID |
| `toipfx` | TOI system prefix | Grouping by system |
| `tfopwg_disp` | TFOP working group disposition | Primary TESS label/status |
| `pl_orbper` | Orbital period, days | Folding/model feature |
| `pl_tranmid` | Transit midpoint | Folding/model feature |
| `pl_trandurh` | Transit duration, hours | Folding/model feature |
| `pl_trandep` | Transit depth | Feature |
| `pl_rade` | Planet radius, Earth radii | Metadata/feature |
| `st_tmag` | TESS magnitude | Quality feature |
| `ra` | Right ascension | Crossmatch/search |
| `dec` | Declination | Crossmatch/search |

Important: TOI labels evolve. Store a download timestamp and source URL with every snapshot.

## Dataset 4: ExoFOP-TESS Public TOI CSV

Use this as an additional public TOI/follow-up snapshot.

Direct CSV:

```text
https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv
```

Download:

```python
import pandas as pd

url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
exofop_toi = pd.read_csv(url)
if exofop_toi.empty:
    raise RuntimeError("ExoFOP-TESS TOI CSV returned no rows")
exofop_toi.to_parquet("data/raw/exofop/tess_toi_public.parquet", index=False)
```

Common columns may include names such as:

```text
TIC ID
TOI
Disposition
Epoch (BJD)
Period (days)
Duration (hours)
Depth (mmag)
Planet Radius (R_Earth)
Planet Insolation (Earth Flux)
```

Do not hard-code only these names. Read the actual CSV header and preserve it.

Required behavior:

- Save the exact downloaded CSV headers to `metadata/source_snapshots.json`.
- Preserve the raw CSV or parquet snapshot with a download timestamp.
- If expected fields are missing, stop and report the actual headers.
- Do not scrape the ExoFOP HTML interface to recover missing CSV fields.

## Dataset 5: TESS Light Curves From MAST

Join key:

```text
TOI tid -> MAST target name "TIC <tid>"
```

Recommended Python path:

```python
from lightkurve import search_lightcurve

def download_tess_lightcurves(tid: int, download_dir: str):
    result = search_lightcurve(f"TIC {tid}", mission="TESS")
    if len(result) == 0:
        raise RuntimeError(f"No TESS light curve search results for TIC {tid}")
    return result.download_all(download_dir=download_dir)
```

Before downloading a batch:

```python
def tess_search_metadata(tid: int) -> list[dict]:
    result = search_lightcurve(f"TIC {tid}", mission="TESS")
    if len(result) == 0:
        return []
    return [dict(row) for row in result.table]
```

Write the search metadata to `metadata/download_manifest.jsonl` before downloading. If a target has many TESS sectors, cap the number of sectors in the first pass and log which sectors were used.

For very large TESS pulls, MAST sector bulk download scripts exist, but do not use them unless the user approves the expected storage footprint first:

```text
https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html
```

TESS light curve FITS structure is broadly similar to Kepler:

| Column | Meaning | Use |
|---|---|---|
| `TIME` | Time coordinate | Time series input |
| `SAP_FLUX` | Raw aperture flux | Raw input option |
| `SAP_FLUX_ERR` | Raw flux uncertainty | Weighting |
| `PDCSAP_FLUX` | Corrected flux | Recommended first ML input |
| `PDCSAP_FLUX_ERR` | Corrected flux uncertainty | Weighting |
| `QUALITY` | Quality bitmask | Mask bad cadences |

## Confirmed Exoplanet Reference Table

Use this for confirmed-planet metadata, not as the main transit-classifier label table.

Table:

```text
pscomppars
```

Download:

```python
confirmed = tap_csv("select * from pscomppars")
confirmed.to_parquet("data/raw/nasa_exoplanet_archive/pscomppars.parquet", index=False)
```

Access validation:

```python
required = {"pl_name", "hostname", "discoverymethod", "disc_year"}
available = tap_columns("pscomppars")
missing = required - available
if missing:
    raise RuntimeError(f"Missing pscomppars columns: {sorted(missing)}")
```

Useful columns:

```text
pl_name
hostname
discoverymethod
disc_year
pl_orbper
pl_rade
pl_bmasse
st_teff
st_rad
st_mass
sy_dist
ra
dec
```

## Exomoon Handling

There is no large confirmed-positive exomoon dataset suitable for standard supervised learning.

Recommended exomoon module:

1. Fit and subtract a planet-only transit model.
2. Extract residual features around each transit.
3. Search for:
   - transit timing variations,
   - transit duration variations,
   - moon-like pre/post-transit residual dips,
   - inconsistent depth/centroid shifts,
   - repeatability across transits.
4. Rank candidates for human review.
5. Evaluate against HEK targets and published candidate systems, but do not treat them as confirmed labels.

Synthetic injection is out of scope for this phase because the user requested no synthetic data.

Allowed for this phase:

- real Kepler/TESS light curves,
- real NASA Exoplanet Archive labels,
- real ExoFOP-TESS public labels/status fields,
- real published HEK/candidate systems as case studies.

Not allowed for this phase:

- injected synthetic planet signals,
- injected synthetic moon signals,
- simulated light curves used as training positives.

## Suggested Repository Layout

```text
data/
  raw/
    nasa_exoplanet_archive/
      koi_cumulative.parquet
      tess_toi.parquet
      pscomppars.parquet
    exofop/
      tess_toi_public.parquet
    mast/
      kepler/
      tess/
  interim/
    joined_labels/
    folded_lightcurves/
  processed/
    train/
    validation/
    test/
  cache_delete_after_training/
metadata/
  source_snapshots.json
  download_manifest.jsonl
scripts/
  download_exoplanet_archive.py
  download_mast_lightcurves.py
  build_training_manifest.py
  preprocess_lightcurves.py
  cleanup_raw_training_data.py
```

## Training Manifest Schema

Create one manifest row per candidate signal, not one row per star.

```text
source_catalog            e.g. koi_cumulative, tess_toi, exofop_toi
mission                   Kepler, TESS
target_id                 kepid or tid
target_id_namespace       KIC or TIC
signal_id                 kepoi_name or toi
system_group_id           kepid, toipfx, or TIC ID
label                     planet_confirmed, planet_candidate_high_confidence, false_positive, weak_candidate, unknown
label_origin              catalog_disposition, human_review
label_source_url          source table URL
label_downloaded_at_utc   timestamp
period_days               orbital period
epoch                     transit epoch
epoch_time_system         BKJD, BJD, BTJD, etc.
duration_hours            transit duration
depth                     transit depth
snr                       model SNR if available
ra_deg                    right ascension
dec_deg                   declination
mag                       Kepler or TESS magnitude
lightcurve_paths          JSON list of local FITS/parquet paths
raw_files_deleted         boolean
raw_redownload_metadata   JSON object with enough info to re-download raw files
split                     train, validation, test
notes                     free text
```

## Leakage Controls

Split by system/target, not by individual light curve file.

Rules:

- All signals from the same `kepid` must stay in the same split.
- All signals from the same TIC ID / TOI prefix should stay in the same split.
- Avoid training on one planet in a multi-planet system and testing on another from the same system.
- Preserve catalog snapshot dates because dispositions change.

## Minimal Python Dependencies

```text
pandas
pyarrow
requests
astropy
astroquery
lightkurve
numpy
scipy
scikit-learn
tqdm
```

Optional:

```text
torch
tensorflow
batman-package
wotan
transitleastsquares
```

## Implementation Checklist

1. Query TAP schemas for `cumulative`, `toi`, and `pscomppars`.
2. Download raw catalog snapshots and save immutable copies.
3. Build normalized labels with provenance.
4. Estimate download size for the first Kepler batch.
5. Confirm the working data directory is below the 100 GB cap.
6. Download Kepler light curves for selected `kepid` values only.
7. Parse FITS files and extract `TIME`, `PDCSAP_FLUX`, flux error, and `QUALITY`.
8. Mask bad cadences.
9. Normalize per sector/quarter.
10. Fold light curves using catalog period and epoch.
11. Build global and local transit views.
12. Split by target/system.
13. Train a Kepler-only baseline classifier/ranker.
14. Evaluate on held-out Kepler.
15. Save model weights, config, metrics, manifest, and processed compact arrays.
16. Delete raw downloaded FITS training files once processed artifacts are verified.
17. Only then evaluate transfer performance on a limited TESS subset.
18. Keep exomoon ranking separate from exoplanet classification.
19. Do not use synthetic data.
20. Stop and ask the user before any step that could exceed about 100 GB of local data.

## Bottom Line

Use Kepler DR25 KOI labels plus a bounded, selected set of MAST Kepler light curves as the primary supervised training dataset. Build a working Kepler classifier/ranker first. Use TESS TOIs plus a limited set of MAST TESS light curves only after the Kepler baseline works. Keep raw training data under about 100 GB, delete raw FITS files after verified processing/training, and preserve manifests so the data can be re-downloaded. Treat exomoons as a real-data residual/anomaly-ranking problem because there is no large confirmed-positive real exomoon dataset. Do not use synthetic data in this phase.
