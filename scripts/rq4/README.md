# RQ4 scripts — Cross-dataset robustness (Table 6, Table 7)

| Script | Purpose | Output |
|---|---|---|
| `run_main_sweep.sh` | 4-encoder × 4-setting × 4-seed sweep on AMLworld + AMLNet | `results/rq4/{amlworld,amlnet}_main_sweep.csv` |

`run_main_sweep.sh` is a thin wrapper around the shared
`scripts/run_main_table.sh` that pins `DATASETS="amlworld amlnet"` and
restricts `ENCODERS` to the four BN-equipped variants used in Table 6
(`bgrl gbt dgi_bn grace_bn`). Per-tag temp files are merged into the
final CSVs by `scripts/merge_main_table.sh`.

Re-running the sweep overwrites the temp files but appends to (not
overwrites) the final per-dataset CSV. To start clean, delete the
final CSV first.
