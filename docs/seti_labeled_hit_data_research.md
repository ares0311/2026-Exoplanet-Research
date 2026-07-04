# Research Note: Real Per-hit Labeled BL/SETI Classification Data for Calibration

Date checked: 2026-07-05

## Agent Instructions

This note is for a coding agent working on calibration of `semisupervised_anomaly_score` / `semisupervised_anomaly_score`-like SETI hit scoring. Treat a dataset as usable calibration ground truth only if it has all of the following:

1. A real downloadable machine-readable table.
2. One row per hit, event, signal-of-interest, or candidate.
3. A human or published-review verdict column that is not merely the model's own prediction.
4. Label categories that can be mapped to `false_positive`, `follow_up`, `insufficient_evidence`, or a documented equivalent.

Do **not** treat aggregate processed-hit counts, plots, paper tables without downloadable rows, target catalogs, unlabeled event summaries, or synthetic/injected signals as calibration labels.

## Required Deliverables for a Coding Agent

The coding agent should produce a metadata-first inventory, not jump straight to downloading raw telescope data.

Recommended output layout:

```text
data/label_inventory/
  seti_source_manifest.csv
  seti_candidate_table_manifest.csv
  seti_rejected_sources.csv
  seti_download_manifest.csv
  seti_label_data_dictionary.csv
  api_access_notes.md
  raw_metadata/
    arxiv/
    seti_berkeley/
    github/
    zenodo/
    journal_pages/
  schema_snapshots/
```

Purpose of each file:

| File | Purpose |
|---|---|
| `seti_source_manifest.csv` | One row per paper/source checked, including title, DOI/arXiv, URLs, date checked, and outcome |
| `seti_candidate_table_manifest.csv` | One row per real downloadable table/file discovered, labeled or unlabeled |
| `seti_rejected_sources.csv` | One row per rejected source, with an explicit rejection reason |
| `seti_download_manifest.csv` | Hashes, byte sizes, URLs, and retrieval timestamps for every downloaded metadata/table file |
| `seti_label_data_dictionary.csv` | Only for accepted labeled datasets; documents original label columns and mappings |
| `api_access_notes.md` | Credentials, rate limits, API endpoints, query examples, and failure modes |
| `raw_metadata/` | Preserved raw API/page responses used to justify decisions |
| `schema_snapshots/` | Column lists and inferred types for every machine-readable table checked |

## Data Inventory Schema

### `seti_source_manifest.csv`

Required columns:

```text
source_id
paper_title
first_author
year
doi
arxiv_id
journal
primary_url
source_type
searched_locations
machine_readable_tables_found
usable_labeled_table_found
date_checked_utc
checked_by
notes
```

Allowed `source_type` values:

```text
paper
project_page
github_repository
zenodo_record
journal_supplement
archive_api
```

### `seti_candidate_table_manifest.csv`

Required columns:

```text
table_id
source_id
table_name
hosting_service
url
file_format
downloaded
local_raw_path
sha256
byte_size
row_count
column_count
columns
has_row_per_hit_or_candidate
has_independent_human_verdict
label_columns
unique_label_values
usable_for_calibration
rejection_reason_if_any
date_checked_utc
```

Allowed `file_format` values:

```text
csv
tsv
fits
hdf5
json
votable
html_table
pdf_only
unknown
```

### `seti_rejected_sources.csv`

Required columns:

```text
source_id
table_id
url
rejection_reason
machine_readable
row_level
has_verdict_label
label_is_independent_of_model
synthetic_or_injected
aggregate_only
notes
date_checked_utc
```

Recommended controlled rejection reasons:

```text
no_machine_readable_table_found
aggregate_counts_only
event_summary_without_verdict_labels
target_catalog_not_signal_labels
model_outputs_not_ground_truth
synthetic_or_injected_only
top_candidate_examples_only
pdf_or_figure_only
unable_to_access_page
```

### `seti_label_data_dictionary.csv`

Only populate this file for accepted labeled datasets. Required columns:

```text
accepted_dataset_id
original_column
original_description
original_dtype
original_unit
original_allowed_values
project_column
project_dtype
project_allowed_values
mapping_rule
mapping_confidence
mapping_notes
```

Allowed project labels:

```text
false_positive
follow_up
insufficient_evidence
candidate
rfi
unknown
```

Do not map source labels into the project schema unless the source label vocabulary is explicitly present in a downloaded table or in the source documentation.

## API and Access Plan

No credentials were found to be required for the public metadata checks in this note. Use API keys only if a service requires them for higher-volume access; do not put secrets in repo files.

| Service | Credential needed? | Use | Notes |
|---|---:|---|---|
| arXiv | No | Paper metadata; PDF/TeX source links | Use arXiv page/API first; ancillary links must be explicit |
| Berkeley SETI pages | No | BL project pages and event CSV links | Metadata first; do not bulk-download raw telescope data |
| Breakthrough Listen Open Data Archive | No public credential found | Search raw/reduced public data | Files may be GB-scale; only query metadata unless explicitly needed |
| GitHub public repos | No for light use | Repository metadata, README, candidate files | Unauthenticated API is rate-limited; optional token for higher rate |
| Zenodo | No for public records | Data records and file metadata | Use record API and file checksums where available |
| Journal pages | Usually no for metadata; varies | Supplementary files and data availability statements | Automated access may be blocked; log blocked pages explicitly |
| NASA Exoplanet Archive TAP | No | Kepler/TESS table discovery | Use `TAP_SCHEMA` before hard-coded table names |
| MAST | No for public metadata/data | Kepler/TESS HLSP and mission product discovery | Use `astroquery.mast` or MAST APIs |
| ExoFOP-TESS | No for public TOI metadata; account may be needed for some features | TOI/disposition ecosystem | Log whether endpoints require interactive/session access |
| ADS | Yes for API token | Literature completeness checks | Optional; use manual/web search if no token |

### arXiv Pull Pattern

For each paper:

1. Save arXiv abstract page metadata.
2. Record full-text links exposed on the arXiv page.
3. If TeX source is downloaded, grep source for:

```text
data availability
supplement
zenodo
github
csv
fits
table
machine-readable
```

4. Reject unless a real table URL/file is found.

### Berkeley SETI Pull Pattern

For project pages such as `https://seti.berkeley.edu/listen2019/opendata.html`:

1. Save raw HTML.
2. Parse all anchors.
3. Record link text and resolved URL in `seti_candidate_table_manifest.csv`.
4. For linked CSVs, download only the CSV headers first if possible.
5. Accept as labeled only if a verdict label column exists.

Do not guess CSV URLs from link text. Parse them from the HTML.

### GitHub Pull Pattern

For public repositories:

1. Save README and repository tree listing.
2. Search filenames for:

```text
label
labels
candidate
candidates
events
hits
rfi
verdict
followup
follow_up
```

3. For each candidate file, record raw URL, byte size, SHA256, row count, and columns.
4. Reject candidate-example folders unless there is a row-level label table.

### Zenodo Pull Pattern

For every candidate Zenodo record:

1. Save record JSON from the Zenodo API.
2. Record DOI, title, authors, publication date, file list, file sizes, checksums, and version.
3. Download only metadata-sized files first.
4. Reject if the record is target catalogs, code, figures, or simulations without real per-hit verdicts.

## Data Volume and Download Budget

The searched sources include raw telescope products that can be enormous. The coding agent must separate **metadata/table pulls** from **raw observation pulls**.

Known volume facts from checked sources:

| Source | Data-volume implication |
|---|---|
| Price et al. 2020 / BL DR1 | arXiv abstract reports all data products total about `219 TB`; do not download raw data for label discovery |
| Berkeley SETI Price event CSVs | Metadata/event-summary CSVs only; expected to be small compared with raw observations |
| Ma et al. 2023 top candidates | Candidate data exist in GitHub/open archive context; inspect metadata first |
| Kepler/TESS labels | Catalog tables are small-to-medium; light curves and pixel products are large and should not be downloaded during label inventory |

Default download policy:

```text
metadata and schemas: allowed
CSV/TSV label candidates under 100 MB: allowed
FITS/HDF5 candidate files: inspect headers/metadata first
raw BL observations: forbidden for this task unless explicitly approved
Kepler/TESS light curves: forbidden for label-inventory task unless explicitly approved
```

For each downloaded file, write:

```text
url
local_path
retrieved_at_utc
http_status
content_type
byte_size
sha256
source_id
table_id
```

## Bottom Line

I did **not** find a larger real per-hit/per-candidate human-labeled SETI/BL table suitable for threshold calibration in the sources checked below.

The closest machine-readable source found is the Price et al. 2020 Berkeley SETI event CSV set, but the published columns are event-summary quantities, not verdict labels. Therefore it is not a labeled ground-truth table for classifier calibration.

Operational conclusion for the project:

- Keep the HIP99427 citizen-science labels as the only currently verified per-hit labels in hand.
- Do not set a global anomaly threshold from those 124 rows.
- Use the sources below only as unlabeled or weakly labeled validation/context unless a future downloadable verdict table is found.

## Source-by-source Findings

| Source | Machine-readable per-row table found? | Human verdict labels found? | Row count usable as labels | Calibration use |
|---|---:|---:|---:|---|
| Enriquez et al. 2017, 692 stars | No usable downloadable labeled table found | No | 0 | Not usable |
| Price et al. 2020, 1327 stars | Event CSVs exist | No verdict column in documented schema | 0 labeled rows | Unlabeled event summaries only |
| Sheikh et al. 2021 / BLC1 | No downloadable candidate-verdict table found | No | 0 | Not usable |
| Smith et al. 2021 / BLC1 companion | No downloadable candidate-verdict table found | No | 0 | Not usable |
| Jacobson-Bell et al. 2025 / arXiv:2411.16556 | No ground-truth labeled evaluation table found | No | 0 | Method paper; not a label source |
| Breakthrough Listen Exotica Catalog | Catalog table exists in paper, but it is target taxonomy | No signal/hit verdicts | 0 | Not a signal-classification dataset |
| Ma et al. 2023 deep-learning GBT search | Repo/top-candidate data exist | Not a broad per-hit human-label table | 0 calibration rows | Candidate examples only |
| Choza et al. 2024 97 galaxies | Published search paper; no labeled table found from checked sources | No | 0 | Not usable |

## 1. Enriquez et al. 2017

Paper:

- Enriquez et al. 2017, "The Breakthrough Listen Search for Intelligent Life: 1.1-1.9 GHz observations of 692 Nearby Stars", ApJ 849, 104.
- arXiv: [1709.03491](https://arxiv.org/abs/1709.03491)
- DOI: [10.3847/1538-4357/aa8d1b](https://doi.org/10.3847/1538-4357/aa8d1b)

Verified facts:

- The arXiv abstract says 11 events passed the thresholding algorithm and detailed analysis found them consistent with anthropogenic RFI.
- The arXiv page lists only PDF and TeX source links; no ancillary machine-readable data table was visible on the arXiv page checked.
- The arXiv metadata says "13 pages, 7 figures, 4 tables", which is not the same as a downloadable per-hit label table.

Finding:

No real downloadable per-hit or per-candidate classification table was found for Enriquez et al. 2017. The paper-level statement that 11 events were consistent with RFI is useful scientifically, but it is not a per-hit labeled training/calibration table.

Calibration verdict:

```text
usable_for_calibration = false
reason = "No downloadable per-row verdict table found."
```

## 2. Price et al. 2020

Paper:

- Price et al. 2020, "The Breakthrough Listen Search for Intelligent Life: Observations of 1327 Nearby Stars over 1.10-3.45 GHz", AJ 159, 86.
- arXiv: [1906.07750](https://arxiv.org/abs/1906.07750)
- DOI: [10.3847/1538-3881/ab65f1](https://doi.org/10.3847/1538-3881/ab65f1)
- Berkeley SETI data page: [Breakthrough Listen: 1327 Star Analysis and Public Data Release 1](https://seti.berkeley.edu/listen2019/opendata.html)

Verified facts:

- The arXiv abstract states that after excluding events consistent with terrestrial RFI, the work was left with zero candidates.
- The Berkeley SETI page confirms three CSV event files exist: L-band Green Bank, S-band Green Bank, and 10CM Parkes.
- The Berkeley SETI page documents the CSV columns as:

```text
Source
DriftRateMax
Nevent
FreqMid
DriftBW
DriftRates
Freqs
FileID
SNR
```

Finding:

Those columns describe event groups and their signal properties. The documented schema does **not** include a verdict or label column such as `candidate`, `RFI`, `rejected`, `follow_up`, `false_positive`, or `insufficient_evidence`.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Machine-readable event summaries exist, but no per-row human verdict label is documented."
```

Coding-agent handling:

- These CSVs may be useful as **unlabeled event examples**.
- Do not map every row to `false_positive` solely because the paper reports zero surviving candidates. That would convert a paper-level conclusion into per-row labels without a published per-row verdict.

## 3. Sheikh et al. 2021 and BLC1 Companion Work

Primary BLC1 analysis paper:

- Sheikh et al. 2021, "Analysis of the Breakthrough Listen signal of interest blc1 with a technosignature verification framework".
- arXiv: [2111.06350](https://arxiv.org/abs/2111.06350)
- DOI: [10.1038/s41550-021-01508-8](https://doi.org/10.1038/s41550-021-01508-8)

Companion detection paper:

- Smith et al. 2021, "A radio technosignature search towards Proxima Centauri resulting in a signal-of-interest".
- arXiv: [2111.08007](https://arxiv.org/abs/2111.08007)
- DOI: [10.1038/s41550-021-01479-w](https://doi.org/10.1038/s41550-021-01479-w)

Verified facts:

- Sheikh et al. state that BLC1 was not an extraterrestrial technosignature and was an electronically drifting intermodulation product of local time-varying interferers.
- Sheikh et al. also report dozens of similar RFI instances at harmonically related frequencies.
- The arXiv page lists PDF and TeX source links, but no visible ancillary machine-readable candidate-verdict table.
- The companion Smith et al. paper reports the BLC1 signal-of-interest and points to the Sheikh et al. companion for analysis.

Finding:

No real downloadable table of all reviewed BLC1-related candidates with per-candidate verdicts was found in the checked arXiv pages or search results.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Scientifically important candidate analysis, but no downloadable per-candidate verdict table found."
```

## 4. Jacobson-Bell et al. 2025 / arXiv:2411.16556

Paper:

- Jacobson-Bell et al. 2025, "Anomaly Detection and Radio-frequency Interference Classification with Unsupervised Learning in Narrowband Radio Technosignature Searches".
- arXiv: [2411.16556](https://arxiv.org/abs/2411.16556)
- DOI: [10.3847/1538-3881/adb8e7](https://doi.org/10.3847/1538-3881/adb8e7)

Verified facts:

- arXiv:2411.16556 is real and relevant.
- The paper presents GLOBULAR, an unsupervised clustering method using HDBSCAN.
- The abstract says it benchmarks against the Choza et al. turboSETI-only search of 97 nearby galaxies, with false-positive hit and event reduction rates.
- The arXiv page lists PDF, experimental HTML, and TeX source links, but no visible ancillary labeled evaluation dataset.

Finding:

No real per-hit human-labeled evaluation table was found. This is a method paper relevant to anomaly reduction, but the checked public metadata does not expose a ground-truth labeled hit table.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Relevant unsupervised method; no downloadable human-labeled ground-truth table found."
```

## 5. Breakthrough Listen Exotica Catalog

Paper:

- Lacki et al. 2021, "One of Everything: The Breakthrough Listen Exotica Catalog", ApJS 257, 42.
- arXiv: [2006.11304](https://arxiv.org/abs/2006.11304)
- DOI: [10.3847/1538-4365/ac168a](https://doi.org/10.3847/1538-4365/ac168a)

Verified facts:

- The arXiv abstract says the Exotica Catalog is a 963-entry collection of 816 distinct targets.
- It contains four target samples: Prototype, Superlative, Anomaly, and Control.
- The purpose is target selection / survey breadth, not classification of detected radio hits.

Finding:

The Exotica Catalog is a target taxonomy and observation-planning/catalog resource. It is not a per-hit or per-candidate SETI signal-classification dataset.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Target catalog, not signal/hit verdict labels."
```

## 6. Other Checked SETI / Technosignature Sources

### Ma et al. 2023 Deep-learning GBT Search

Sources:

- Ma et al. 2023, "A deep-learning search for technosignatures of 820 nearby stars".
- arXiv: [2301.12670](https://arxiv.org/abs/2301.12670)
- DOI: [10.1038/s41550-022-01872-z](https://doi.org/10.1038/s41550-022-01872-z)
- Berkeley SETI project page: [Deep Learning Search for Technosignatures](https://seti.berkeley.edu/ml_gbt/)
- GitHub repository: [PetchMa/ML_GBT_SETI](https://github.com/PetchMa/ML_GBT_SETI)

Verified facts:

- The Berkeley SETI page states that follow-up observations did not re-detect the signals, so they do not pass criteria for bona fide technosignature candidates.
- The same page says the GitHub repo contains full target lists and data for the top candidates.
- The GitHub README states it contains top 8 candidates and their original data.

Finding:

This is useful for qualitative review of top candidate examples. It is not a broad per-hit labeled ground-truth table. The available top-candidate data should not be treated as a calibrated binary label set.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Top-candidate examples exist, but not a broad per-hit human-label table."
```

### Choza et al. 2024 Nearby-galaxy Search

Source:

- Choza et al. 2024, "The Breakthrough Listen Search for Intelligent Life: Technosignature Search of 97 Nearby Galaxies".
- arXiv: [2312.03943](https://arxiv.org/abs/2312.03943)
- DOI: [10.3847/1538-3881/acf576](https://doi.org/10.3847/1538-3881/acf576)

Verified facts:

- The paper reports a turboSETI narrowband Doppler drift search of 97 nearby galaxies.
- It removed RFI using ON/OFF cadence filtering and zero-drift rejection.
- The arXiv page lists PDF and TeX source links, but no visible ancillary labeled hit table.

Finding:

No per-hit/per-candidate human-verdict table was found in the checked public metadata.

Calibration verdict:

```text
usable_for_calibration = false
reason = "Published search and benchmark context, but no downloadable verdict table found."
```

## Do-not-use-as-label Rules

A coding agent must not create labels by inference from paper-level conclusions:

| Tempting shortcut | Why it is invalid |
|---|---|
| Label every Price et al. event row as `false_positive` because the paper reports zero surviving candidates | The CSV schema does not document per-row verdicts; this would fabricate row labels |
| Label BLC1-related RFI examples without a downloaded verdict table | The papers analyze BLC1 and RFI families, but no per-candidate verdict table was found |
| Use Exotica Catalog `Anomaly` targets as signal anomalies | `Anomaly` is a target-selection category, not a detected-signal label |
| Use GLOBULAR cluster IDs as ground truth | Cluster assignments from an unsupervised method are model outputs, not independent human labels |
| Use simulated/injected signals as real positives | They are useful for sensitivity/completeness tests, not real technosignature labels |

## Recommended Calibration Path

Given the current evidence, threshold calibration should be staged:

1. Keep the HIP99427 human-review dataset as the only verified per-hit label set currently available.
2. Use it only for sanity checks and qualitative ordering, not global threshold selection.
3. Use unlabeled BL event tables and hit tables for distributional checks, not supervised calibration.
4. Build a larger internal review set by sampling high-, medium-, and low-score hits across multiple targets and asking humans to label them with the same schema.
5. Version the review table with immutable columns:

```text
hit_id
source_file
target
frequency_hz
drift_rate_hz_s
snr
score
review_label
reviewer_id
review_timestamp_utc
review_notes
paper_or_dataset_source
```

Minimum useful calibration target:

```text
>= 1,000 reviewed rows total
>= 50 follow_up or candidate-like rows, if naturally present
multiple targets
multiple observing bands
multiple score deciles
```

If the positive class remains rare, use precision-at-k and review-budget calibration instead of a fixed binary threshold.

## Accepted-dataset Schema If a Future Label Table Is Found

If a future search finds a valid per-row SETI label table, normalize it into:

```text
accepted_dataset_id
source_id
row_id
original_row_id
target
observation_id
file_id
frequency_hz
drift_rate_hz_s
snr
band
telescope
pipeline
original_label
project_label
label_source
human_reviewed
review_protocol
source_url
source_sha256
notes
```

Required validation:

1. `row_id` is unique.
2. `project_label` is one of the controlled project labels.
3. `original_label` is preserved exactly.
4. `label_source` states whether the label came from a paper table, citizen review, official archive field, or author repository.
5. `human_reviewed` must be `true` for threshold-calibration use unless the project explicitly accepts another independent label source.
6. Rows with model-predicted labels must be marked `human_reviewed=false` and excluded from ground-truth calibration.

## Exhaustive-search Protocol for Future SETI Label Hunting

Run this checklist before declaring the search exhausted in the future:

1. Search arXiv for: `"Breakthrough Listen" "candidate" "table"`, `"turboSETI" "events" "csv"`, `"SETI" "RFI" "labeled dataset"`, `"technosignature" "candidate" "Zenodo"`, `"Breakthrough Listen" "GitHub" "candidates"`.
2. For each relevant paper, open the arXiv page and check whether full-text links include anything beyond PDF/TeX source.
3. Check the journal DOI page for "supplementary material", "machine-readable table", or "data availability".
4. Check Berkeley SETI project pages under `https://seti.berkeley.edu/` for paper-specific data pages.
5. Check GitHub orgs/users:
   - `UCBerkeleySETI`
   - `PetchMa`
   - author-linked repositories from each paper page
6. Check Zenodo by exact paper title and author names.
7. Reject sources unless a downloaded file contains a per-row independent verdict column.
8. Save a rejection log with URL, date, reason, and whether a machine-readable but unlabeled table exists.

## Minimal Implementation Pseudocode

```python
for source in sources:
    save_source_metadata(source)
    links = discover_machine_readable_links(source)
    for link in links:
        record = inspect_link_without_bulk_download(link)
        save_schema_snapshot(record)
        if not record.machine_readable:
            reject(record, "no_machine_readable_table_found")
            continue
        if not record.row_level:
            reject(record, "aggregate_counts_only")
            continue
        if not record.has_label_column:
            reject(record, "event_summary_without_verdict_labels")
            continue
        if record.labels_are_model_outputs:
            reject(record, "model_outputs_not_ground_truth")
            continue
        accept(record)
```

Acceptance requires positive evidence. Absence of evidence is a rejection, not a guess.

## Comprehensive Protocol: Ensuring All Labeled Kepler and TESS Datasets Have Been Found

This section is separate from SETI hit labels. It is included because the project also uses Kepler/TESS labeled data for exoplanet/anomaly work.

Goal:

```text
Create a reproducible inventory of every official or publication-backed Kepler, K2, and TESS table that contains row-level labels/dispositions usable for supervised learning.
```

Official sources to audit:

| Source | Why it matters |
|---|---|
| NASA Exoplanet Archive | Official Kepler/KOI/TCE/confirmed-planet tables and API access |
| MAST | Mission products, DV reports, TESS/Kepler/K2 HLSPs |
| ExoFOP-TESS | TOI and follow-up disposition ecosystem |
| VizieR/CDS | Journal-published machine-readable tables |
| ADS/arXiv | Paper trail and supplementary-data discovery |
| GitHub/Zenodo linked from papers | Author-hosted labeled datasets |

Minimum label-like fields to search for:

```text
disposition
disp
vetting
vett
label
class
classification
candidate
confirmed
false_positive
fp
planet_candidate
pc
toi_disposition
tfopwg_disp
koi_disposition
tce_disposition
robovetter
```

### Step A: NASA Exoplanet Archive TAP Inventory

Use the Exoplanet Archive TAP service, not a hard-coded table list, to discover live tables.

Base TAP endpoint:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync
```

Discovery query:

```sql
SELECT
  table_name,
  description
FROM TAP_SCHEMA.tables
WHERE
  LOWER(table_name) LIKE '%kepler%'
  OR LOWER(table_name) LIKE '%koi%'
  OR LOWER(table_name) LIKE '%tce%'
  OR LOWER(table_name) LIKE '%k2%'
  OR LOWER(table_name) LIKE '%tess%'
  OR LOWER(table_name) LIKE '%toi%'
```

Example URL pattern:

```text
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<URL_ENCODED_ADQL>&format=csv
```

Column scan query for every returned table:

```sql
SELECT
  table_name,
  column_name,
  description,
  unit
FROM TAP_SCHEMA.columns
WHERE table_name = '<TABLE_NAME>'
ORDER BY column_name
```

Label-value enumeration pattern for each candidate label column:

```sql
SELECT
  <LABEL_COLUMN>,
  COUNT(*) AS n
FROM <TABLE_NAME>
GROUP BY <LABEL_COLUMN>
ORDER BY n DESC
```

Accept a table as labeled only if:

1. It contains a row-level disposition/classification column.
2. The label vocabulary is documented or can be enumerated from the table itself.
3. The source is official archive, peer-reviewed supplementary data, or author-linked release.

### Step B: Known Official Starting Points to Verify Live

Do not assume these table names remain current. Verify each through `TAP_SCHEMA` before querying.

| Mission | Candidate official table family | Label/disposition concept to verify |
|---|---|---|
| Kepler | KOI cumulative catalog | KOI disposition / false positive / candidate / confirmed |
| Kepler | DR24 KOI/TCE tables | Robovetter or pipeline disposition |
| Kepler | DR25 KOI/TCE tables | Robovetter disposition and reliability/completeness-related labels |
| K2 | K2 planet/candidate tables | Candidate/confirmed/false-positive dispositions |
| TESS | TOI table | TOI disposition / TFOP working-group disposition |
| TESS | TESS SPOC TCE table family, if present | TCE vetting/disposition fields |
| TESS | Confirmed planets table subset | Confirmed planet labels, not candidate vetting labels |

### Step C: MAST HLSP Inventory

Search MAST HLSP records for Kepler/K2/TESS products with label-like terms:

```text
Kepler false positive
Kepler disposition
Kepler robovetter
Kepler TCE
K2 candidates false positive
TESS TOI disposition
TESS TCE vetting
TESS planet candidates catalog
TESS QLP candidates
TESS SPOC TCE
```

For each HLSP:

1. Record HLSP name, DOI or documentation URL, mission, and data-product URL.
2. Download only metadata/index files first.
3. Inspect columns for label-like fields.
4. Confirm row counts and unique label values.
5. Reject if it only has light curves/images and no row-level labels.

Suggested MAST access methods:

```python
from astroquery.mast import Observations

# Discover public mission products by collection/project/keyword.
obs = Observations.query_criteria(
    obs_collection=["Kepler", "K2", "TESS"],
)

# Then inspect products, but do not bulk download light curves for the label audit.
products = Observations.get_product_list(obs[:10])
```

Use MAST for product discovery and metadata, not as the primary source of disposition labels unless a table/product explicitly contains row-level labels.

### Step D: VizieR and Literature Audit

Search VizieR/CDS by exact paper titles and bibcodes for major Kepler/TESS catalogs. Required query targets include:

```text
Kepler DR24 catalog
Kepler DR25 catalog
Kepler Robovetter
TESS Objects of Interest
TESS Input Catalog
TESS planet candidates
K2 planet candidates
Planet Hunters Kepler
Planet Hunters TESS
```

For every VizieR hit:

1. Confirm catalog identifier from the VizieR page.
2. Pull the `ReadMe`.
3. List every table and its row count.
4. Scan columns for disposition/label fields.
5. Download only tables with row-level labels.

Suggested VizieR machine-readable pattern:

```text
https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=<CONFIRMED_CATALOG_ID>&-out.max=unlimited&-out=*
```

Do not invent VizieR identifiers. Confirm every catalog ID from the VizieR page or ReadMe before using it.

### Step E: Data Dictionary Rules for Kepler/TESS

For every accepted Kepler/TESS labeled source, create a source-specific data dictionary with:

```text
source_id
archive
table_name
column_name
column_description
unit
dtype
nullable
is_label_column
unique_values_if_label
project_mapping
notes
```

Keep original mission labels separate from project labels. Example pattern:

```text
original_label_column = koi_disposition
original_label_values = CANDIDATE | FALSE POSITIVE | CONFIRMED
project_label_mapping = documented explicitly in inventory, not inferred silently
```

If the label source is TESS TOI disposition, keep `tfopwg_disp` or equivalent original values intact and map them only after enumerating the live unique values.

### Step F: Acceptance Criteria for the Kepler/TESS Label Audit

The audit is complete only when the coding agent has produced:

```text
data/label_inventory/kepler_tess_label_sources.csv
data/label_inventory/rejected_kepler_tess_sources.csv
data/label_inventory/source_schema_snapshots/
```

Required columns for `kepler_tess_label_sources.csv`:

```text
source_name
mission
archive
table_name_or_file
url
doi_or_bibcode
row_count
label_column
unique_label_values
label_mapping_to_project_schema
date_checked_utc
verification_method
notes
```

Required columns for `rejected_kepler_tess_sources.csv`:

```text
source_name
mission
archive
url
date_checked_utc
rejection_reason
machine_readable
has_row_level_labels
notes
```

The coding agent must fail the audit if:

- A known official table family has not been checked through live metadata.
- A source is accepted without a row count.
- A source is accepted without enumerating unique label values.
- A label mapping is inferred without documenting the original label vocabulary.

## Final Recommendation

For SETI/BL calibration, do not proceed as if a broad public ground-truth label set exists. The checked BL/SETI literature provides event summaries, top-candidate examples, target catalogs, and unsupervised methods, but not a larger real human-labeled per-hit table suitable for calibrating a high-score threshold.

The next defensible step is to create a project-owned human review set from real hits, while continuing to log future literature/data searches using the rejection protocol above.
