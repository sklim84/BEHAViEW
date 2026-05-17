# Main table sweep results

Re-run of Table 3 (`tab:rq1`) and Table 6 (`tab:rq4`) under the current
code revision (loss dispatch + final_loss tracking). The earlier sweep
written by `scripts/_backup/run_main_table_original.sh` lives under
`results/_backup/exp_results_{hofinet_ab,amlworld,amlnet}.csv`.

| File | Coverage | Expected rows |
|---|---|---|
| `hofinet.csv` | 9 encoders × 4 settings (a/b/c/d) × 4 seeds | 144 |
| `amlworld.csv` | 4 encoders × 4 settings × 4 seeds | 64 |
| `amlnet.csv` | 4 encoders × 4 settings × 4 seeds | 64 |

Encoder set:
- HOFINET: `gbt bgrl dgi mvgrl grace dgi_bn mvgrl_bn grace_bn gin`
- Cross-dataset: `bgrl gbt dgi_bn grace_bn`

Run via `scripts/run_main_table.sh` (per-GPU dispatches write to
`.tmp_${DS}_${TAG}.csv`); merge with `scripts/merge_main_table.sh`.

Loss is `BootstrapLatent`, matching the unified BYOL bootstrap used in
the Table 9 (`tab:loss_table`) BYOL column. The earlier sweep passed
`--loss BarlowTwins`, which the previous code silently ignored but the
current code respects, hence the explicit change.
