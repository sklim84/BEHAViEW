# RQ1 scripts — Homophily Recovery (Table 3)

| Script | Output |
|---|---|
| `run_main_sweep.sh` | `results/rq1/main_sweep.csv` (after `scripts/merge_main_table.sh`) |

`run_main_sweep.sh` is a thin wrapper over `scripts/run_main_table.sh`
that pins `DATASETS=atnet` and the full nine-encoder set used in
Table 3 (`gbt bgrl dgi mvgrl grace dgi_bn mvgrl_bn grace_bn gin`).
Parallel dispatches write per-tag temp files; `merge_main_table.sh`
collapses them into `main_sweep.csv`.
