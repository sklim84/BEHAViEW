# BehaView results

CSV files are organized by the research question (RQ) they feed.

```
results/
├── rq1/                        # Homophily Recovery (Table 3, Figure 1)
│   ├── main_sweep.csv          # 9 encoders × 4 settings × 4 seeds × HOFINET
│   └── case_study/             # Figure 1 fixtures
├── rq2/                        # Signal Preservation (Table 5)
│   ├── paired_t_test.csv       # Holm-Bonferroni results (long format)
│   ├── paired_t_test_bn.csv
│   └── pool_variants.csv       # HAP / Cycle pool alternatives
├── rq3/                        # Label Efficiency (Figure 6, Table 8)
│   ├── behaview_{hofinet,amlworld,amlnet}.csv
│   ├── supervised_{hofinet,amlworld,amlnet}.csv   # GCN/GAT/SAGE/MLP/XGB/LGBM/CARE-GNN
│   ├── bwgnn_gaga_{...}.csv
│   ├── caregnn_pcgnn_{...}.csv
│   ├── consisgad_{...}.csv
│   └── boosting_behav.csv
├── rq4/                        # Cross-dataset Robustness (Table 6, Table 7)
│   ├── amlworld_main_sweep.csv
│   └── amlnet_main_sweep.csv   # (pending Step 9 re-run)
├── appendix/                   # Tables 9, 10
│   ├── feature_ablation.csv
│   └── loss_ablation/          # 4 logical CSVs for Table 9
└── _backup/                    # original sweep CSVs, safe runs, loss_ablation_raw
```

Each producer script is listed in the matching subdirectory's
`README.md` and writes back to the same path on rerun.
