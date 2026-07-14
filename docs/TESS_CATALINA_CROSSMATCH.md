# TESS–Catalina Variability-Label Crossmatch

Version 0.2.62 adds the bounded gate between the verified Catalina label source
and the 2,790-TIC cached TESS representation inventory. Its immutable contract
is `metadata/tess_catalina_crossmatch_contract_v1.json`; its implementation is
`Skills/crossmatch_tess_catalina_labels.py`.

## Authorized scope

Only a deterministic 216-TIC pilot is authorized. The pilot hashes every TIC
with the frozen seed, selects before sharding, and then partitions by
`tic_id % 6`. This produces about 36 TICs per process shard. Each shard uses
six MAST batch workers with six exact TIC IDs per request, so the one-terminal
supervisor exercises six shards × six workers without opening six terminals.
Requests return only ID, J2000 position, T/V magnitudes, Gaia ID, proper motion,
duplicate ID, and object type.

The 47,055-row Catalina table is not sharded: it is one 1,166,660-byte shared
read-only input. The first shard acquires it under a filesystem lock, verifies
SHA-256 `7b0d497f…aaeb60`, and atomically promotes it into the ignored in-repo
cache. Other shards wait and then verify the same file. Downloading six copies
would waste time and violate shared-input ownership.

Version 0.2.64 parses both measured source row shapes: 44,538 unflagged rows
end at the mandatory 71-byte core, while 2,517 flagged rows include the
optional byte-73 inspection flag. Missing trailing flag bytes mean no flag;
rows outside the supported 71- to 73-byte fixed-width range still fail closed
before any MAST request.
The version 0.2.64 release gate passed 2,760 default tests plus Ruff/mypy as
8/8 supervised gates in 34.3 seconds under the canonical 6×6 topology.

## Match safeguards

- Both source positions are treated as ICRS/J2000. TIC proper motion is
  retained for audit, not silently applied without a separately pinned epoch.
- Search within 3 arcseconds; accept only at or below 1 arcsecond.
- Require exactly one Catalina candidate around the TIC.
- Require Catalina V versus TIC V within 2 magnitudes; if TIC V is absent,
  allow the deliberately broader Catalina V versus TESS T limit of 5.
- Reject TIC duplicate entries, non-stellar TIC objects, and Catalina blend
  flag `f`.
- Preserve the exact Catalina class code and flag. The derived
  `benchmark_family` never replaces the raw publication label.
- Require global one-to-one Catalina-source reconciliation across all six
  outputs before any future label use.

Every pilot row remains `role=training` with `training_authorized=false`.
No FITS file is opened, no embedding is extracted, and no model is trained.

## Merged execution sequence

After version 0.2.62 is merged and `main` is clean, first validate all commands
and the storage projection:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python Skills/run_six_shards.py \
  --script crossmatch_tess_catalina_labels.py \
  --dry-run \
  --expected-new-gb 0.01 \
  -- \
  --max-targets 216 \
  --batch-size 6
```

Then run the measured pilot from the same supervisor:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_six_shards.py \
  --script crossmatch_tess_catalina_labels.py \
  --expected-new-gb 0.01 \
  -- \
  --max-targets 216 \
  --batch-size 6
```

Compare aggregate TIC/s, retries, missing rows, timeouts, accepted/rejected
matches, and duplicate Catalina source IDs. A clean rate and zero service
errors may justify a separate contract revision for all 2,790 TICs. Low
overlap, throttling, or sub-linear scaling is a stop signal, not permission to
relax the precommitted scientific safeguards.
