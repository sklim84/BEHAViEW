# Loss ablation results

Each CSV here is one logical experiment for the BehaView loss-substitution
narrative (Appendix `app:loss_table` and surrounding text).

| File | Experiment | Rows |
|---|---|---|
| `byol_substitution.csv` | 9 encoders × BYOL bootstrap × setting (d) × ATNET × 4 seeds. Same-environment BYOL column for Table 9. | 36 |
| `native_loss.csv` | Each encoder × its paper-native contrastive loss × setting (d) × {ATNET, AMLworld, AMLNet} × 4 seeds. Native column for Table 9 and cross-dataset note. | 52 |
| `loss_robustness.csv` | GBT × {BootstrapLatent, BarlowTwins, InfoNCE, JSD} × setting (d) × ATNET × 4 seeds. BehaView loss-robustness sentence in Appendix. | 16 |
| `view_similarity.csv` | DGI+BN × 4 losses × {(a), (b)} × ATNET × 4 seeds. Table `tab:loss_ab_check` for the "low loss at (a) is structural" paragraph. | 32 |

Original step-by-step CSVs (step1.csv ... step7_p2.csv) are preserved
in `results/_backup/loss_ablation_raw/` for traceability. The merge logic
that produces the four files above is documented inline in the relevant
Python aggregation snippet (no separate merge script: regenerating these
files is a one-off task).
