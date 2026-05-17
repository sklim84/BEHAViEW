# Appendix scripts

| Script | Output |
|---|---|
| `run_feature_ablation.sh` | `results/appendix/feature_ablation.csv` |

The loss-ablation experiments in `results/appendix/loss_ablation/` are
historical: the per-step sweeps that produced them are archived under
`scripts/_backup/run_loss_ablation_*.sh`. To rerun, copy the relevant
backup script and point its `--metric_save_path` to the matching CSV
in `results/appendix/loss_ablation/`.
