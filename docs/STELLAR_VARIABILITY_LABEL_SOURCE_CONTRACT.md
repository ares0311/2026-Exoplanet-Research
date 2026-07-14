# Stellar-Variability Label Source Contract

Version 0.2.60 pins the first publication-backed stellar-variability label
source required by the Phase 3 representation comparison. The immutable
contract is `metadata/stellar_variability_label_source_contract_v1.json`; run
`Skills/verify_stellar_variability_label_source.py` to compare it against the
current primary-source metadata.

## Accepted source

The source is Drake et al. (2014), *ApJS* 213, 9, distributed by CDS/VizieR as
`J/ApJS/213/9/table3` (catalog DOI
[`10.26093/cds/vizier.22130009`](https://doi.org/10.26093/cds/vizier.22130009)).
It contains 47,055 machine-readable periodic-variable rows with sky position,
period, amplitude, one of 17 published class codes, and optional inspection
flags. The compressed table is 1,166,660 bytes.

This source passes the project's label-source protocol because it provides
row-level machine-readable records and publication-backed classifications,
including explicit inspection/ambiguity flags. It is not assumed perfect: the
paper reports a few-percent ambiguity between RRc and contact binaries, and the
future benchmark must retain individual class identities rather than treating
all entries as one interchangeable negative class.

## Verification boundary

The verifier performs exactly five bounded operations:

1. HEAD the compressed CDS table and verify length, ETag, modification time,
   and content type.
2. Query VizieR `TAP_SCHEMA.columns` and verify the required schema.
3. Verify the 47,055-row count.
4. Verify all 17 class counts, whose sum must equal the row count.
5. Read three TAP sample rows and validate identifiers, coordinates, periods,
   and class codes.

It prints progress/ETA, writes a structured artifact and Run Report, and fails
closed on drift. It downloads zero full-catalog payload bytes and does not
resolve TIC coordinates, crossmatch targets, open light curves, extract
embeddings, train a model, or alter production scoring. Five small dependent
checks finish below the three-minute threshold, so this gate is intentionally
sequential.

## Rejected alternatives

- Gaia DR3 `vari_classifier_result` is automated classifier output, not an
  independent human/publication verdict. It may later be auxiliary metadata,
  but it is not ground truth for this gate.
- StarEmbed is a valuable expert-vetted benchmark, but its Hugging Face corpus
  is gated and approximately 160 GB. It requires access approval and exceeds
  the project's 100 GB pre-go-live storage ceiling.

## Next gate

After merged-code verification succeeds, design a bounded match between these
published labels and the 2,790 TICs in the committed cached-TESS representation
inventory. Resolve only metadata first, define angular/magnitude/duplicate and
target-group safeguards, and measure a small live-service batch. The full
independent-TIC pass must then use the single-parent six-shard/six-worker shape,
with disjoint outputs, progress/ETA, storage preflight, and one aggregate Run
Report. Crossmatch and training remain unauthorized until that design and its
evidence are committed.

## Merged verification evidence

The merged verifier passed all 5 operations on 2026-07-14 in 3.334 seconds.
It verified 47,055 total rows, every one of the 17 pinned class counts, all
required columns and datatypes, the 1,166,660-byte compressed delivery
metadata, and three labeled sample rows. It downloaded zero full-catalog
payload bytes. The durable artifact is
`artifacts/manifests/stellar_variability_label_source_verification_v1.json`
with SHA-256
`eb5d4bc6ae02065752e515fff19ed9b012d163f1d82a2be958796a65ba339b9a`;
Run Report commit `b0003bb`. This closes source identity only. Version 0.2.61
does not authorize crossmatch, embedding extraction, or training.
