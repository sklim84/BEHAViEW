# RQ3 scripts — Label efficiency

| Script | Output | Notes |
|---|---|---|
| `run_behaview.sh` | `results/rq3/behaview_${ds}.csv` | BehaView at {1%, 5%, 10%} labels; encoder picked per dataset (BGRL/GBT) |
| `run_supervised.sh` | `results/rq3/supervised_${ds}.csv` | GCN, GAT, GraphSAGE, MLP, XGBoost, LightGBM, CARE-GNN |
| `run_bare_mlp.sh` | append to `results/rq3/supervised_${ds}.csv` | bare MLP (Linear+ReLU only); replaces modern MLP rows |
| `run_bwgnn_gaga.sh` | `results/rq3/bwgnn_gaga_${ds}.csv` | BWGNN + GAGA |
| `run_caregnn_pcgnn.sh` | `results/rq3/caregnn_pcgnn_${ds}.csv` | CARE-GNN + PC-GNN |
| `run_consisgad.sh` | `results/rq3/consisgad_${ds}.csv` | ConsisGAD |
| `run_boosting_behav.sh` | `results/rq3/boosting_behav.csv` | XGBoost/LightGBM with behavioral-only features |

`run_additional_exp.sh` (still at top level, Phase D will split) writes
the HOFINET BehaView label-fraction part to
`results/rq3/behaview_hofinet.csv`; its feature-ablation part belongs
to the appendix.

Re-running any of the above writes to the exact CSV path documented
here (no temp suffix, no env-var override required).
