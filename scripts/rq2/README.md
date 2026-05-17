# RQ2 scripts — Signal Preservation (Table 5, Figure rq2_bn)

| Script | Output |
|---|---|
| `compute_paired_t_test.py` | `results/rq2/paired_t_test.csv` |
| `run_pool_variants.sh` | `results/rq2/pool_variants.csv` |

`compute_paired_t_test.py` reads
`results/rq1/main_sweep.csv` (HOFINET) and
`results/rq4/amlworld_main_sweep.csv` (AMLworld), so it stays in sync
with whatever `scripts/rq1/run_main_sweep.sh` and
`scripts/rq4/run_main_sweep.sh` last produced.

`run_pool_variants.sh` runs HAP / Cycle pool alternatives on HOFINET
(4 encoders × {c, d} × pool variant × 4 seeds = 64 runs).
