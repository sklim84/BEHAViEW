# RQ4 results — Cross-dataset robustness

Backbone of Table 6 (`tab:rq4`) and Table 7 (`tab:transfer`).

| File | Source | Coverage |
|---|---|---|
| `amlworld_main_sweep.csv` | `scripts/rq4/run_main_sweep.sh` (DATASETS=amlworld) | 10 encoders × 4 settings × 4 seeds (current data spans all 10 encoders; Table 6 uses 4 BN-equipped) |
| `amlnet_main_sweep.csv` | `scripts/rq4/run_main_sweep.sh` (DATASETS=amlnet) | pending Step 9 re-run (no canonical AMLNet CSV in the current repo state) |
| `transfer.csv` | TBD (Phase D) | Table 7 transfer-learning data |

Auxiliary sweeps that previously fed Table 6 lived under the temp
`_safe_${encoder}.csv` names and now sit in
`results/_backup/safe_runs/` for traceability. The four-setting
columns of Table 6 are reproducible from `amlworld_main_sweep.csv`
plus the pending `amlnet_main_sweep.csv`.
