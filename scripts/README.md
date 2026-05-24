# BehaView scripts

Scripts are organized by the research question (RQ) they support, so
each script lives next to its peers and its output path mirrors that
RQ in `results/`.

```
scripts/
├── run_main_table.sh           # shared 4-setting × encoder sweep (RQ1 + RQ4)
├── merge_main_table.sh         # per-tag tmp file collapser for the above
├── rq1/                        # Homophily Recovery (Table 3)
│   └── run_main_sweep.sh       # ATNET wrapper -> results/rq1/main_sweep.csv
├── rq2/                        # Signal Preservation (Table 5)
│   ├── compute_paired_t_test.py
│   └── run_pool_variants.sh
├── rq3/                        # Label Efficiency (Figure 6, Table 8)
│   ├── run_behaview.sh         # BehaView at {1%, 5%, 10%} labels, all 3 datasets
│   ├── run_supervised.sh       # GCN, GAT, GraphSAGE, MLP, XGBoost, LightGBM, CARE-GNN
│   ├── run_bwgnn_gaga.sh       # BWGNN, GAGA
│   ├── run_caregnn_pcgnn.sh    # CARE-GNN, PC-GNN
│   ├── run_consisgad.sh        # ConsisGAD
│   └── run_boosting_behav.sh   # XGBoost/LightGBM with behavioral-only features
├── rq4/                        # Cross-dataset Robustness (Table 6, Table 7)
│   └── run_main_sweep.sh       # AMLworld+AMLNet wrapper -> results/rq4/
├── appendix/                   # Tables 9, 10
│   └── run_feature_ablation.sh
└── _backup/                    # archived scripts (older naming, sweep history)
```

Each script's docstring documents the canonical output path; rerunning
it produces the same CSV at the same path, so consumers (analysis
scripts and the paper) do not need to be updated.
